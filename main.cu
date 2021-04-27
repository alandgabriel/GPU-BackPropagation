
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include<fstream>
#include <sstream>
#include <vector>
#define HX 558 //Filas de X  train:558, total:699
#define WX 9 //Columnas de X
#define HY 558 //Filas de Y train:558, total:699
#define HXtest 141
#define HYtest 141
#define FILE_NAME_TRAIN "/home/alan/Documents/GPUs_Deeplearning_IIMAS/Cuda NNS/breast-cancer-train.csv"
#define FILE_NAME_TEST "/home/alan/Documents/GPUs_Deeplearning_IIMAS/Cuda NNS/breast-cancer-test.csv"
#define NUM_ITER 2000
#define alphamax 0.5
#define NUM_MODELOS 100
#define TILE_W 10 //Tamaño mosaicos
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
using namespace std;

void MatMul(float *a,float *d,float *c,int ha,int wa,int wb);
void generateRandom(float *h_a,int rows, int cols);
void leerDatos(float *X,float *y,string file);
void printMatrix(float *h_a, int rows, int cols);
void sigmoid(float *a,float *c, int w,int h);
float costoCE(float *a,float *p);
void VecAddition(float *a,float *c,float *res);
void VecSubtraction(float *a,float *c,float *res,int width);
void VecScalAddition(float *a,float c,int size);
void MatScalMul(float *a,float m,int rows);
void MatTranspose(float *a,float *at,int width,int height);
float VecSum(float *a);
void predict(float *datos,float *pesos,float bias,float *preds,int h,int w);
float score(float *preds, float *y_r,int w);
int buscaMejorModelo(float *a);
void train_serial();
void train_paralelo();
void parallTranspose(float *a,float *at);
void parallMult(float *a,float *d,float *c,int ha,int wa,int wb);
void parallPredict(float *datos,float *pesos,float bias,float *preds,int h,int w);
void parallActiv (float *a,float *d,float *c,int ha,int wa,int wb);

float astep = alphamax/NUM_MODELOS;

// global timers
double serialTimer = 0.0;
float parallelTimer = 0.0;

float *X,*y,*W,*Jc,*XT,*y_pred,*Xtest,*ytest,*y_predtest,*modelos,*alphas, *d_X, *d_W, *d_XT;
int sizeX=HX*WX*sizeof(float );
int sizeXtest=HXtest*WX*sizeof(float );
int sizeY=HY*sizeof(float );
int sizeYtest=HYtest*sizeof(float );
int sizeW=WX*sizeof(float );
int sizeJ=NUM_ITER* sizeof(float);
int sizeMod=NUM_MODELOS* sizeof(float);
__device__ float d_b;

__global__ void childOptimizer(float * d_X, float*d_XT, float*d_W ){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < NUM_ITER) {

    }
}

//Kernel padre para entrenamiento de los modelos
__global__ void parallTrain(float *d_X,float *d_XT, float *d_W){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < NUM_MODELOS){
        d_b= 0;
        int b = ceil((float)NUM_ITER / 1024.0) + 1;
        int t = 1024;
        childOptimizer <<<b, t >>>( d_X, d_XT, d_W);
        cudaDeviceSynchronize();


    }


}


__global__ void activFunc (float *d_a, float *d_d, float *d_c, int ha, int wa, int wb) {
    int i = threadIdx.y + (blockIdx.y * blockDim.y);
    int j = threadIdx.x + (blockIdx.x * blockDim.x);
    float temp = 0;
    if (i < ha && j < wb) {
        for (int k = 0; k < wa; k++) {
            temp += d_a[i * wa + k] * d_d[j + (wb * k)];
        }
        d_c[i * wb + j] = temp;

    }
    cudaDeviceSynchronize();
}
// Kernel multilplicacion matrices hilo computa 1 elemento de C
__global__ void onelementMtxMult (float *d_a, float *d_d, float *d_c, int ha, int wa, int wb){
    int i = threadIdx.y + (blockIdx.y * blockDim.y);
    int j = threadIdx.x + (blockIdx.x * blockDim.x);
    float temp = 0;
    if (i < ha && j < wb){
        for (int k =0; k < wa; k++){
            temp += d_a [i * wa + k] * d_d [j + (wb * k)];
        }
        d_c [i * wb + j] = temp;
    }

}

// Kernel MatrixTranspose
__global__ void PTranspose(float *d_a,float *d_at){
    int i = threadIdx.y + (blockIdx.y * blockDim.y);
    int j = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i<HX && j<WX){
        d_at[j*HX+i] = d_a[i*WX+j];

    }
}



int main() {
    //Datos para entrenar
    X = (float *) malloc(sizeX);
    XT = (float *) malloc(sizeX);
    y = (float *) malloc(sizeY);
    y_pred = (float *) malloc(sizeY);
    y_predtest = (float *) malloc(sizeYtest);
    W = (float *) malloc(sizeW);
    Jc = (float *) malloc(sizeJ);
    //Datos de test
    Xtest = (float *) malloc(sizeXtest);
    ytest = (float *) malloc(sizeYtest);
    //Datos de los modelos
    modelos = (float *) malloc(sizeMod);
    alphas = (float *) malloc(sizeMod);

    leerDatos(X,y,FILE_NAME_TRAIN);
    leerDatos(Xtest,ytest,FILE_NAME_TEST);



//Proceso Paralelo
    train_paralelo();

    y_pred = (float *) malloc(sizeY);
    y_predtest = (float *) malloc(sizeYtest);
    W = (float *) malloc(sizeW);
    Jc = (float *) malloc(sizeJ);

    //Datos de los modelos
    modelos = (float *) malloc(sizeMod);
    alphas = (float *) malloc(sizeMod);

    //Proceso serial
    clock_t start = clock();
    train_serial();
    clock_t end = clock();
    serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
    cout << "Tiempo del proceso serial: " << serialTimer << endl;


    free(X), free(y); free(W),free(y_pred),free(Xtest),free(ytest),free(y_predtest);
    return 0;
}

void MatMul(float *a,float *d,float *c,int ha,int wa,int wb){
    float sum;
    for(int i= 0; i<ha; i++) {
        for(int j=0;j<wb; j++){
            sum = 0;
            for(int k=0;k<wa; k++){
                sum+=a[i*wa+k]*d[j+(wb*k)];
            }
            c[i*wb+j] = sum;
        }
    }
}


void printMatrix(float *h_a, int rows, int cols){
    for(int i=0; i<rows; i++ ){
        for(int j=0; j<cols; j++){
            cout<<h_a[i*cols+j]<<" ";
        }
        cout<<endl;
    }
}

void generateRandom(float *h_a,int rows, int cols){
    // Initialize seed
    srand(time(NULL));
    for(int i=0; i<rows*cols; i++){
        h_a[i] = rand() % 10 + 1;
    }
}

void leerDatos(float *X,float *y,string file){
    ifstream archivo(file);
    string linea;
    char delimiter = ',';
    getline(archivo,linea);
    int i=0;
    while (getline(archivo,linea)){
        stringstream stream(linea);
        string num,code,grosor_tumor, tam_celula, form_celula, adhesion, celula_epit, nucleos,cromatina,nucleos_normales,mitosis,clase;

        getline(stream,num,delimiter);
        //getline(stream,code,delimiter);
        getline(stream,grosor_tumor,delimiter);
        getline(stream,tam_celula,delimiter);
        getline(stream,form_celula,delimiter);
        getline(stream,adhesion,delimiter);
        getline(stream,celula_epit,delimiter);
        getline(stream,nucleos,delimiter);
        getline(stream,cromatina,delimiter);
        getline(stream,nucleos_normales,delimiter);
        getline(stream,mitosis,delimiter);
        getline(stream,clase,delimiter);

        X[i*WX] = stof(grosor_tumor);
        X[i*WX+1] = stof(tam_celula);
        X[i*WX+2] = stof(form_celula);
        X[i*WX+3] = stof(adhesion);
        X[i*WX+4] = stof(celula_epit);
        X[i*WX+5] = stof(nucleos);
        X[i*WX+6] = stof(cromatina);
        X[i*WX+7] = stof(nucleos_normales);
        X[i*WX+8] = stof(mitosis);
        y[i] = stof(clase);

//        cout<<"Fila: "<<i<<endl;
//        cout<<X[i*WX]<<" ";
//        cout<<X[i*WX+1]<<" ";
//        cout<<X[i*WX+2]<<" ";
//        cout<<X[i*WX+3]<<" ";
//        cout<<X[i*WX+4]<<" ";
//        cout<<X[i*WX+5]<<" ";
//        cout<<X[i*WX+6]<<" ";
//        cout<<X[i*WX+7]<<" ";
//        cout<<X[i*WX+8]<<" ";
//        cout<<y[i]<<endl;
//        cout<<"\n"<<endl;
        i++;
    }
    archivo.close();
}

void sigmoid(float *a,float *c, int w,int h){
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            c[i*w+j] = 1/(1+exp(-a[i*w+j]));
        }
    }
}

float costoCE(float *a,float *p){
    float suma=0;
    for(int i=0;i<HY;i++){
        suma +=  p[i]*log(a[i])+(1-p[i])*log(1-a[i]);
    }
    return float (-(1.0/HY))*suma;
}

void VecAddition(float *a,float *c,float *res){
    for(int i= 0; i<HY; i++) {
        res[i] = a[i] + c[i];
    }
}
void VecScalAddition(float *a,float c,int size){
    for(int i= 0; i<size; i++) {
        a[i] = a[i] + c;
    }
}

float VecSum(float *a){
    float suma=0;
    for(int i= 0; i<HY; i++) {
        suma += a[i];
    }
    return suma;
}

void VecSubtraction(float *a,float *c,float *res,int width){
    for(int i= 0; i<width; i++) {
        res[i] = a[i] - c[i];
    }
}
void MatScalMul(float *a,float m,int rows){
    for (int i=0;i<rows;i++){
        a[i] = a[i]*m;
    }
}

void MatTranspose(float *a,float *at,int width,int height){
    for(int j=0;j<width;j++){
        for(int i=0;i<height;i++){
            at[j*height+i] = a[i*width+j];
        }
    }
}

void predict(float *datos,float *pesos,float bias,float *preds,int h,int w){
    float *Z,*S;
    Z = (float *) malloc(h*sizeof(float ));
    S = (float *) malloc(h*sizeof(float ));
    MatMul(datos,pesos,Z,h,w,1);
    VecScalAddition(Z,bias,h);
    sigmoid(Z,S,1,h);
    for(int i=0;i<h;i++){
        if(S[i]<=0.5){
            preds[i] = 0;
        }
        else{
            preds[i] = 1;
        }
    }
    free(Z),free(S);
}

void parallPredict(float *datos,float *pesos,float bias,float *preds,int h,int w){
    float *Z,*S;
    Z = (float *) malloc(h*sizeof(float ));
    S = (float *) malloc(h*sizeof(float ));
    parallMult(datos,pesos,Z,h,w,1);
    VecScalAddition(Z,bias,h);
    sigmoid(Z,S,1,h);
    for(int i=0;i<h;i++){
        if(S[i]<=0.5){
            preds[i] = 0;
        }
        else{
            preds[i] = 1;
        }
    }
    free(Z),free(S);
}

float score(float *preds, float *y_r,int w){
    float aciertos = 0;
    for(int i=0;i<w;i++){
        if(preds[i]==y_r[i]){
            aciertos+=1;
        }
    }
    return (aciertos/float (w))*100;
}

void generarAlphas(float *a,float step){
    for(int i=0;i<NUM_MODELOS;i++){
        a[i] = float (i)*step;
    }
}

int buscaMejorModelo(float *a){
    float maximo = a[0];
    int maxindex=0;
    for(int i=1;i<NUM_MODELOS;i++){
        if(a[i]>maximo){
            maximo = a[i];
            maxindex = i;
        }
    }
    return maxindex;
}

void train_serial() {
    MatTranspose(X,XT,WX,HX);
    generarAlphas(alphas, astep);
    for (int model = 0; model < NUM_MODELOS; model++) {
        float b = 0;
        for (int iter = 0; iter < NUM_ITER; iter++) {
            float *S, *Sr, cost, *dW, *Z;
            S = (float *) malloc(sizeY);
            Sr = (float *) malloc(sizeY);
            dW = (float *) malloc(sizeW);
            Z = (float *) malloc(sizeY);

            MatMul(X, W, Z, HX, WX, 1);
            VecScalAddition(Z, b, HY);
            sigmoid(Z, S, 1, HY);
            cost = costoCE(S, y);
            //cout << "Iteración " << iter + 1 << " : " << cost << endl;
            Jc[iter] = cost;
            VecSubtraction(S, y, Sr, HY);
            MatMul(XT, Sr, dW, WX, HX, 1);
            MatScalMul(dW, 1.0 / HY, WX);
            float db = VecSum(Sr);
            db = db / HY;

            MatScalMul(dW, alphas[model], WX);
            db = alphas[model] * db;

            VecSubtraction(W, dW, W, WX);
            b = b - db;

            free(dW), free(S), free(Sr), free(Z);
        }
        predict(Xtest, W, b, y_predtest, HXtest, WX);
        float st = score(y_predtest, ytest, HYtest);
        //cout << "Score del modelo test: " << st << "%" << endl;
        predict(X, W, b, y_pred, HX, WX);
        float s = score(y_pred, y, HY);
        //cout << "Score del modelo entrenamiento: " << s << "%" << endl;
        modelos[model] = st;
    }
    int index = buscaMejorModelo(modelos);
    cout << "Mejor score obtenido: " << modelos[index] << endl;
    cout << "Alpha del mejor modelo: " << alphas[index] << endl;
}

void train_paralelo() {

    clock_t start = clock();
    parallTranspose(X,XT);
    cudaMalloc((void **)&d_X, sizeX);
    cudaMalloc((void **)&d_XT, sizeX);
    cudaMalloc((void **)&d_W, sizeW);
    // Transferir datos de host a device
    cudaMemcpy(d_X, X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_XT, XT, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, sizeW, cudaMemcpyHostToDevice);
    int blocks = ceil(NUM_MODELOS / 1024) + 1;
    int threads = 1024;

  //  cudaEvent_t start, stop;

    // events to take time
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start,0);

    // Launch kernel
    parallTrain<<<blocks, threads>>>(d_X,d_XT, d_W);
    generarAlphas(alphas, astep);
    for (int model = 0; model < NUM_MODELOS; model++) {
        float b = 0;
        for (int iter = 0; iter < NUM_ITER; iter++) {
            float *S, *Sr, cost, *dW, *Z;
            S = (float *) malloc(sizeY);
            Sr = (float *) malloc(sizeY);
            dW = (float *) malloc(sizeW);
            Z = (float *) malloc(sizeY);

            parallMult(X, W, Z, HX, WX, 1);
            VecScalAddition(Z, b, HY);
            sigmoid(Z, S, 1, HY);
            cost = costoCE(S, y);
            //cout << "Iteración " << iter + 1 << " : " << cost << endl;
            Jc[iter] = cost;
            VecSubtraction(S, y, Sr, HY);
            parallMult(XT, Sr, dW, WX, HX, 1);
            MatScalMul(dW, 1.0 / HY, WX);
            float db = VecSum(Sr);
            db = db / HY;

            MatScalMul(dW, alphas[model], WX);
            db = alphas[model] * db;

            VecSubtraction(W, dW, W, WX);
            b = b - db;

            free(dW), free(S), free(Sr), free(Z);
        }
        parallPredict(Xtest, W, b, y_predtest, HXtest, WX);
        float st = score(y_predtest, ytest, HYtest);
        //cout << "Score del modelo test: " << st << "%" << endl;
        parallPredict(X, W, b, y_pred, HX, WX);
        float s = score(y_pred, y, HY);
        //cout << "Score del modelo entrenamiento: " << s << "%" << endl;
        modelos[model] = st;
    }
    int index = buscaMejorModelo(modelos);
    cout << "Mejor score obtenido: " << modelos[index] << endl;
    cout << "Alpha del mejor modelo: " << alphas[index] << endl;

    clock_t end = clock();
    parallelTimer = double (end-start) / double(CLOCKS_PER_SEC);
    cout << "Tiempo del proceso paralelo: " << parallelTimer << endl;
}



void parallTranspose(float *a,float *at){
    float * d_a, * d_at;
    cudaMalloc((void **)&d_a, sizeX);
    cudaMalloc((void **)&d_at, sizeX);

    cudaMemcpy(d_a, a, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_at, at, sizeX, cudaMemcpyHostToDevice);


    dim3 block(32, 32);
    auto gridx = WX/block.x +1;
    auto gridy = HX/block.y +1;

    dim3 grid(gridx,gridy );


    PTranspose<<<grid, block>>>(d_a, d_at);
    // cudaThreadSynchronize ();



    // Copy data from device to host
    cudaMemcpy(at, d_at, sizeX, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree (d_at);
/*
    for (int i =0; i< sizeX ; i++){
        cout << a[i]<< " ";
    }
    cout <<endl;
    for (int i =0; i< sizeX ; i++){
        cout << at[i]<< " ";
    }
    cout <<endl;*/


}

void parallActiv (float *a,float *d,float *c,int ha,int wa,int wb){
    float *d_a,*d_d,*d_c;

    cudaMalloc((void **)&d_a, sizeX);
    cudaMalloc((void **)&d_d, wa*sizeof(float));
    cudaMalloc((void **)&d_c, ha*sizeof(float));



    cudaMemcpy(d_a, a, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, wa*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, ha*sizeof(float), cudaMemcpyHostToDevice);




    dim3 block(32, 32);
    auto gridx = wb/block.x +1;
    auto gridy = ha/block.y +1;

    dim3 grid(gridx,gridy);


    activFunc<<<grid, block>>>(d_a, d_d, d_c,ha,wa,wb);
    // cudaThreadSynchronize ();



    // Copy data from device to host
    cudaMemcpy(c, d_c, ha*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_d);
    cudaFree(d_c);
}


void parallMult(float *a,float *d,float *c,int ha,int wa,int wb){
    float *d_a,*d_d,*d_c;

    cudaMalloc((void **)&d_a, sizeX);
    cudaMalloc((void **)&d_d, wa*sizeof(float));
    cudaMalloc((void **)&d_c, ha*sizeof(float));



    cudaMemcpy(d_a, a, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, wa*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, ha*sizeof(float), cudaMemcpyHostToDevice);




    dim3 block(32, 32);
    auto gridx = wb/block.x +1;
    auto gridy = ha/block.y +1;

    dim3 grid(gridx,gridy);


    onelementMtxMult<<<grid, block>>>(d_a, d_d, d_c,ha,wa,wb);

    // cudaThreadSynchronize ();



    // Copy data from device to host
    cudaMemcpy(c, d_c, ha*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_d);
    cudaFree(d_c);
/*
    for (int i =0; i< ha ; i++){
        cout << c[i]<< " ";
    }
    cout <<endl;
*/
}