
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include<fstream>
#include <sstream>
#include <vector>
#define HX 699 //Filas de X
#define WX 9 //Columnas de X
#define HY 699 //Filas de Y
#define FILE_NAME "/home/icasasola/proyecto2/breast-cancer-clean.csv"
#define NUM_ITER 100
#define TILE_W 10 //Tamaño mosaicos
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
using namespace std;

void MatMul(float *a,float *d,float *c,int ha,int wa,int wb);
void generateRandom(float *h_a,int rows, int cols);
void leerDatos(float *X,float *y);
void printMatrix(float *h_a, int rows, int cols);
void sigmoid(float *x,float *z);
float costoCE(float *a,float *p);
void VecAddition(float *a,float *c,float *res);
void VecSubtraction(float *a,float *c,float *res);
void VecScalAddition(float *a,float c,float *res);
void MatScalMul(float *a,float m);
void MatTranspose(float *a,float *at,int width,int height);
float VecSum(float *a);

float *X,*y,*W,*Jc,*XT;
int sizeX=HX*WX*sizeof(int);
int sizeY=HY*sizeof(int);
int sizeW=WX*sizeof(int);
int sizeJ=NUM_ITER* sizeof(int);
float alpha = 0.01;

int main() {
    X = (float *) malloc(sizeX);
    XT = (float *) malloc(sizeX);
    y = (float *) malloc(sizeY);
    W = (float *) malloc(sizeW);
    Jc = (float *) malloc(sizeJ);
    float b = 0;

    leerDatos(X,y);
    MatTranspose(X,XT,WX,HX);
    cout<<"q"<<endl;
    for(int iter=0;iter<NUM_ITER;iter++){
        float *S,*Sa,*Sr,cost,*dW,*Z;
        S = (float *) malloc(sizeY);
        Sa = (float *) malloc(sizeY);
        Sr = (float *) malloc(sizeY);
        dW = (float *) malloc(sizeW);
        Z = (float *) malloc(sizeY);
        MatMul(X,W,Z,HX,WX,1);
        sigmoid(Z,S);
        VecScalAddition(S,b,Sa);
        cost = costoCE(Sa,y);
        cout<<"Iteración "<<iter+1<<" : "<<cost<<endl;
        Jc[iter] = cost;
        VecSubtraction(Sa,y,Sr);
        MatMul(XT,Sr,dW,WX,HX,1);
        MatScalMul(dW,1.0/HY);
        float db = VecSum(Sr);
        db = db/HY;

        MatScalMul(dW,alpha);
        db=alpha*db;

        VecSubtraction(W,dW,W);
        b = b - db;
        printMatrix(W,WX,1);
        cout<<"b: "<<b<<endl;

        free(dW),free(S),free(Sa),free(Sr),free(Z);
    }

    free(X), free(y); free(W);
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

void leerDatos(float *X,float *y){
    ifstream archivo(FILE_NAME);
    string linea;
    char delimiter = ',';
    getline(archivo,linea);
    int i=0;
    while (getline(archivo,linea)){
        stringstream stream(linea);
        string num,code,grosor_tumor, tam_celula, form_celula, adhesion, celula_epit, nucleos,cromatina,nucleos_normales,mitosis,clase;

        getline(stream,num,delimiter);
        getline(stream,code,delimiter);
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

void sigmoid(float *x,float *z){
    for(int i=0;i<HX;i++){
        for(int j=0;j<WX;j++){
            z[i*WX+j] = 1/(1+exp(-x[i*WX+j]));
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
void VecScalAddition(float *a,float c,float *res){
    for(int i= 0; i<HY; i++) {
            res[i] = a[i] + c;
    }
}

float VecSum(float *a){
    float suma=0;
    for(int i= 0; i<HY; i++) {
            suma += a[i];
    }
    return suma;
}

void VecSubtraction(float *a,float *c,float *res){
    for(int i= 0; i<HY; i++) {
            res[i] = a[i] - c[i];
    }
}
void MatScalMul(float *a,float m){
    for (int i=0;i<HY;i++){
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