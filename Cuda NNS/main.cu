
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
#define FILE_NAME_TRAIN "/home/icasasola/proyecto2/breast-cancer-train.csv"
#define FILE_NAME_TEST "/home/icasasola/proyecto2/breast-cancer-test.csv"
#define NUM_ITER 2000
#define alphamax 0.5
#define NUM_MODELOS 500
#define NUM_HILOS 25
#define TILE_W 10 //Tamaño mosaicos
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
using namespace std;

void MatMul(float *a,float *d,float *c,int ha,int wa,int wb);
void generateRandom(float *h_a,int rows, int cols);
void compareVectors(float *parallel, float *serial, int rows, int cols);
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
void generarAlphas(float *a,float step);
int buscaMejorModelo(float *a);
void train_serial();
void train_parallel();

//Salto de alpha
float astep = alphamax/NUM_MODELOS;

// global timers
double serialTimer = 0.0;
float parallelTimer = 0.0;

float *X,*y,*W,*Jc,*XT,*y_pred,*Xtest,*ytest,*y_predtest,*modelos,*alphas,*modelos_par;
float *d_mod,*d_alps,*d_X,*d_W,*d_XT,*d_y,*d_Xtest,*d_ytest;
int sizeX=HX*WX*sizeof(float );
int sizeXtest=HXtest*WX*sizeof(float );
int sizeY=HY*sizeof(float );
int sizeYtest=HYtest*sizeof(float );
int sizeW=WX*sizeof(float );
int sizeJ=NUM_ITER* sizeof(float);
int sizeMod=NUM_MODELOS* sizeof(float);


__device__ void MatMulPar(float *a,float *d,float *c,int ha,int wa,int wb){
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

__device__ void sigmoidPar(float *a,float *c, int w,int h){
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            c[i*w+j] = 1/(1+exp(-a[i*w+j]));
        }
    }
}

__device__ float costoCEPar(float *a,float *p){
    float suma=0;
    for(int i=0;i<HY;i++){
        suma +=  p[i]*log(a[i])+(1-p[i])*log(1-a[i]);
    }
    return float (-(1.0/HY))*suma;
}

__device__ void VecScalAdditionPar(float *a,float c,int size){
    for(int i= 0; i<size; i++) {
            a[i] = a[i] + c;
    }
}

__device__ float VecSumPar(float *a){
    float suma=0;
    for(int i= 0; i<HY; i++) {
            suma += a[i];
    }
    return suma;
}

__device__ void VecSubtractionPar(float *a,float *c,float *res,int width){
    for(int i= 0; i<width; i++) {
            res[i] = a[i] - c[i];
    }
}

__device__ void MatScalMulPar(float *a,float m,int rows){
    for (int i=0;i<rows;i++){
        a[i] = a[i]*m;
    }
}

__device__ void predictPar(float *datos,float *pesos,float bias,float *preds,int h,int w){
    float *Z,*S;
    Z = (float *) malloc(h*sizeof(float ));
    S = (float *) malloc(h*sizeof(float ));
    MatMulPar(datos,pesos,Z,h,w,1);
    VecScalAdditionPar(Z,bias,h);
    sigmoidPar(Z,S,1,h);
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

__device__ float scorePar(float *preds, float *y_r,int w){
    float aciertos = 0;
    for(int i=0;i<w;i++){
        if(preds[i]==y_r[i]){
            aciertos+=1;
        }
    }
    return (aciertos/w)*100;
}

__device__ int buscaMejorModeloPar(float *a){
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

__global__ void train_model(float *mod, float *alps, float *dat, float *datT,float *label,float *datTest,float *labelTest) {

        int model = threadIdx.x + (blockIdx.x * blockDim.x);
        if (model<NUM_MODELOS) {
            float b = 0;
            float *WP;
            float *label_pred;
            label_pred = (float *) malloc(HYtest * sizeof(float));
            WP = (float *) malloc(WX * sizeof(float));

            for (int iter = 0; iter < NUM_ITER; iter++) {
                float *S, *Sr, cost, *dW, *Z;
                S = (float *) malloc(HY * sizeof(float));
                Sr = (float *) malloc(HY * sizeof(float));
                dW = (float *) malloc(WX * sizeof(float));
                Z = (float *) malloc(HY * sizeof(float));

                MatMulPar(dat, WP, Z, HX, WX, 1);
                VecScalAdditionPar(Z, b, HY);
                sigmoidPar(Z, S, 1, HY);
                cost = costoCEPar(S, label);
                //cout << "Iteración " << iter + 1 << " : " << cost << endl;
                //Jc[iter] = cost;
                VecSubtractionPar(S, label, Sr, HY);
                MatMulPar(datT, Sr, dW, WX, HX, 1);
                MatScalMulPar(dW, 1.0 / HY, WX);
                float db = VecSumPar(Sr);
                db = db / HY;

                MatScalMulPar(dW, alps[model], WX);
                db = alps[model] * db;

                VecSubtractionPar(WP, dW, WP, WX);
                b = b - db;

                free(dW), free(S), free(Sr), free(Z);
            }
            predictPar(datTest, WP, b, label_pred, HXtest, WX);
            float st = scorePar(label_pred, labelTest, HYtest);

            mod[model] = st;
            free(label_pred);
            free(WP);
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
    modelos_par = (float *) malloc(sizeMod);

    //Reservar memoria en device
    cudaMalloc((void **)&d_X, sizeX);
    cudaMalloc((void **)&d_Xtest, sizeX);
    cudaMalloc((void **)&d_y, sizeY);
    cudaMalloc((void **)&d_ytest, sizeY);
    cudaMalloc((void **)&d_XT, sizeX);
    cudaMalloc((void **)&d_mod, sizeMod);
    cudaMalloc((void **)&d_alps, sizeMod);

    leerDatos(X,y,FILE_NAME_TRAIN);
    leerDatos(Xtest,ytest,FILE_NAME_TEST);
    MatTranspose(X,XT,WX,HX);

    generarAlphas(alphas, astep);

    // Transferir datos de host a device
    cudaMemcpy(d_alps, alphas, sizeMod, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xtest, Xtest, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ytest, ytest, sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_XT, XT, sizeX, cudaMemcpyHostToDevice);

    //Proceso serial
    clock_t start = clock();
    train_serial();
    clock_t end = clock();
    serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
    cout << "Tiempo del proceso serial: " << serialTimer << endl;

    //Proceso en paralelo
    train_parallel();
    //compareVectors(modelos_par,modelos,NUM_MODELOS,1);

    cout << "Speed-up: " << serialTimer / (parallelTimer /1000)<< "X"<<endl;
    cout << "\n"<<endl;

    cudaFree(d_alps),cudaFree(d_X),cudaFree(d_Xtest),cudaFree(d_y),cudaFree(d_ytest);
    cudaFree(d_XT), cudaFree(d_mod);
    free(X), free(y); free(W),free(y_pred),free(Xtest),free(ytest),free(y_predtest);
    free(modelos),free(alphas);
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

void compareVectors(float *parallel, float *serial, int rows, int cols){

    int diff = 0;
    for(int i= 0; i<rows*cols; i++) {
        if (parallel[i] != serial[i]) {
            diff++;
            //cout<<i<<". "<<parallel[i] << " " << serial[i] << "\n" << endl;
        }
    }

    if(diff>0){
        cout<< diff <<" elements different" << endl;
    }
    else
        cout << "Vectors are equal!..." << endl;
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
        a[i] = float (i+1)*step;
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
    //cout<<"Costo:"<<endl;
    //printMatrix(Jc,NUM_ITER,1);
    int index = buscaMejorModelo(modelos);
    cout<<"Mejor score obtenido: "<<modelos[index]<<endl;
    cout<<"Alpha del mejor modelo: "<<alphas[index]<<endl;
}

void train_parallel(){

    int blocks = ceil(NUM_MODELOS / NUM_HILOS) + 1;

    // define timers
    cudaEvent_t start, stop;

    // events to take time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    CUDA_CALL( cudaGetLastError() );
    // Launch kernel
    train_model<<<blocks, NUM_HILOS>>>(d_mod,d_alps,d_X,d_XT,d_y,d_Xtest,d_ytest);
    CUDA_CALL( cudaGetLastError() );

    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaMemcpy(modelos_par, d_mod, sizeMod, cudaMemcpyDeviceToHost);
    int index = buscaMejorModelo(modelos_par);
    cout<<"Mejor score obtenido: "<<modelos_par[index]<<endl;
    cout<<"Alpha del mejor modelo: "<<alphas[index]<<endl;

    cudaEventElapsedTime(&parallelTimer, start, stop);

    cout<< "Tiempo del proceso en paralelo: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;

    // Copy data from device to host
    //cudaMemcpy(modelos_par, d_mod, sizeMod, cudaMemcpyDeviceToHost);
}