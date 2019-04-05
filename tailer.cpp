
#include <stdio.h>
#include <string.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <time.h>
#include <math.h>

//-----------#include "cvconfig.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//----------#include "opencv2/ts.hpp"
//-------------------------
//include per il cv :: resize

//#include <opencv2\core\core.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>

//-------------------------
#include <thread>
#include <mutex>
#include <windows.h>
#include <direct.h>

//include per cv::cuda::resize

#include <opencv2\cudawarping.hpp>
#include <fstream>

using namespace cv;
using namespace std;


//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------



/**
   Struct per passare i dati al thread, contiene:
   --  numero di elementi da scrivere nella memoria fisica
   --  il path in cui scrivere il fotogramma
   --  l'oggetto, mat, contenente il fotogramma
*/

typedef struct{
	int n;//numero di posizioni del vettore 
	Mat *foto;//messaggio
	string *indirizzo;//path in cui andare a scrivere i fotogrammi
} lista;

//----------------------------DICHIARAZIONI VARIABILI GLOBALI---------------------------------


//NOME DELLA CARTELLA CHE CONTIENE LE TILES
String root = "C:\\tail\\";

//dichiaro il semaforo
mutex mu;

//tipo dato contenente le tail che saranno scritte in memoria fisica dal thread
lista da_scrivere;


/**
  * @desc questa funzione è richiamata al momento della creazione del thread.
		  Scrive i fotogrammi nella memoria fisica 
		  I fotogrammi sono salvati nella variabile globale "da_scrivere"
		  per evitare problemi tra produttore e consumatore si utilizza un semaforo.
		  
  * @param void *param tipo dato necessario per il passaggio di dati al thread
  
  * @return la funzione non restituisce risultato
*/

void scrivi_disco(void *param){
	
	lista *copia;
	/*creo una copia della variabile  "da_scrivere"
	poichè il main potrebbe scrivere le tail nella variabile da_scrivere e quindi si creano dei problemi.

	in questo modo ogni thread ha la sua copia di tail da dover scrivere, indipendente senza il rischio di sovvrapposizioni tra computazioni
	*/
	mu.lock();
	copia = (lista*)malloc(sizeof(lista));
	memcpy(copia, &da_scrivere, sizeof(da_scrivere));
	mu.unlock();

	int i;

	for (i = 0; i <copia->n; i++){
		//cout << "ciclo : " << i << endl;
		//imwrite(dato->indirizzo[i], dato->foto[i]);
		
		imwrite(copia->indirizzo[i], copia->foto[i]);
		
	}

	free(copia);
	return;
}




/**
  * @desc effettua la riduzione di una immagine lavorando con il CPU
		  Carica l'immagine dalla memoria fisica e ne effettua il ridimensionamento
		  utilizzando la cv::resize, che effettua una computazione nel CPU
		  		  
  * @param string path : contiene il path dell'immagine da elaborare
  
  * @param float fatt_rid : indica il fattore di riduzione, è un numero decimale minore di 1
							ad esempio con un fattore di 0.5 l'immagine viene ridotta della metà
  
  * @param int metodo : indica il tipo di interpolazione tra i pixel durante la riduzione 
						dell'immagine. 
						0 = INTER_NEAREST , 1= INTER_LINEAR, 2 = INTER_CUBIC
  
  * @return restituisce l'oggetto di tipo "Mat" che contiene l'immagine ridimensionata 
*/

Mat resizeCPU(string path, float fatt_rid, int metodo){
	//LEGGO L'IMMAGINE
	Mat out, immag = imread(path, CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);

	//RIDIMENSIONOP L'IMMAGINE
	cv::resize(immag, out, Size(), fatt_rid, fatt_rid, metodo);

	//restituisco il file ridimensionato
	return out;
}



/**
  * @desc effettua la riduzione di una immagine lavorando con la GPU (scheda grafica)
		  Carica l'immagine dalla memoria fisica e ne effettua il ridimensionamento
		  utilizzando la cuda::resize, che effettua una computazione nella scheda grafica
		  
  * @param string path : contiene il path dell'immagine da elaborare
  
  * @param float fatt_rid : indica il fattore di riduzione, è un numero decimale minore di 1
							ad esempio con un fattore di 0.5 l'immagine viene ridotta della metà
  
  * @param int metodo : indica il tipo di interpolazione tra i pixel durante la riduzione 
						dell'immagine. 
						0 = INTER_NEAREST , 1= INTER_LINEAR, 2 = INTER_CUBIC
  
  * @return restituisce l'oggetto di tipo "Mat" che contiene l'immagine ridimensionata 
*/
Mat resizeGPU(string path, float fatt_rid, int metodo){

	//LEGGO L'IMMAGINE DA PATH
	Mat inputCpu = imread(path, CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);

	//PASSO DALLA RAM L'IMMAGINE APPENA CARICATA ALLA SCHEDA VIDEO
	cuda::GpuMat input(inputCpu);

	//CREO IL FILE DI OUTPUT
	cuda::GpuMat output;

	//RIDIMENSIONO L'IMMAGINE
	cuda::resize(input, output, Size(), fatt_rid, fatt_rid, metodo); // RIDUCO DI fatt_rid VOLTE la X e Y (larghezza e altezza)

	//CONVERTO IL FILE DI OUTPUT PER SALVARLO IN RAM
	Mat out(output);//converto l'oggetto della scheda video in uno nella RAM


	return out;
}



/**
  * @desc effettua la riduzione di un'immagine e consente di scegliere se:
			- lavorare con la scheda video 
			- lavorare con il processore
		  		  
  * @param string path : contiene il path dell'immagine da elaborare
  
  * @param float fatt_rid : indica il fattore di riduzione, è un numero decimale minore di 1
							ad esempio con un fattore di 0.5 l'immagine viene ridotta della metà
  
  * @param int metodo : indica il tipo di interpolazione tra i pixel durante la riduzione 
						dell'immagine. 
						0 = INTER_NEAREST , 1= INTER_LINEAR, 2 = INTER_CUBIC
						
  * @param bool cpu : seleziona se lavorare con il CPU o GPU, se la variabile è settata a :
					  - TRUE : utilizza il CPU
					  - FALSE : utilizza la scheda grafica (GPU)
					
  
  * @return restituisce l'oggetto di tipo "Mat" che contiene l'immagine ridimensionata 
*/

Mat ridimensionaImg(string path, float fatt_rid, int metodo, bool cpu){
	Mat out;
	if ((metodo >= 0) && (metodo <= 3)){
		if (cpu){
			//true--->CPU
			out = resizeCPU(path, fatt_rid, metodo);
			// salvo nella cartella "C:\\tail\\fatt_rid\\"
			root = root + to_string(fatt_rid) + "\\";
			//creo la cartella
			_mkdir(root.c_str());

		}else{
			//false--->SCHEDA GRAFICA (GPU)
			out = resizeGPU(path, fatt_rid, metodo);
			// salvo nella cartella "C:\\tail\\fatt_rid\\"
			root = root + to_string(fatt_rid) + "\\";
			_mkdir(root.c_str());
		}
	}

	//cout << "fine resize" << endl;

	return out;
}


/**
  * @desc prende in input un'immagine allocata nell'oggetto di tipo "Mat", la spezzetta in tails di altezza H e larghezza W
		  e restituisce il vettore di stringhe contenente i path fisici dei fotogrammi ritagliati.
		  Per salvare i ritagli (tiles) in memoria fisica viene utilizzato un thread.
		  
  * @param Mat img : oggetto contenente l'immagine da spezzettare in tiles
  
  * @param int h : altezza della tiles
  
  * @param int w : larghezza della tiles
  
  * @return *string : vettore contenente i path fisici delle tiles calcolate
*/
string* tailer(Mat img, int h, int w){
	

	Mat tail;//oggetto che conterrà la sottoimmagine di larghezza standard
	Mat tmp;//oggetto che contiene il ritaglio di immagine che potrebbe essere più piccolo della tail
	
	

	//alloco l'oggetto tail
	tail = Mat(Size(w, h), img.type());
	tail = cv::Mat::zeros(tail.size(), tail.type());

	int div_col, div_rig;//indicano quante tail (di larghezza prefissata) posso ricavare dalla dimensione del fotogramma originale 
	//cout << "larghezza " << img.cols << endl;
	//cout << "altezza " << img.rows << endl;
	//cout << "altezza tail h " << h << "      larghezza tail w" << w << endl;

	div_col = img.cols / w;//numero di suddivisioni per la larghezza
	div_rig = img.rows / h;//numero di suddivisioni per l' altezza

	/*
	calcolo il resto della divisione
	*/

	int resto_col = img.cols % w;
	int resto_rig = img.rows % h;

	int dim = ((div_col + (resto_col == 0 ? 0 : 1)) * (div_rig + (resto_rig == 0 ? 0 : 1)));

	string* paths = new string[dim];//vettore contenente gli indirizzi fisici delle tails

	//DICHIARO LE VARIABILI DELLA STRUCT CHE PASSO AL THREAD
	da_scrivere.n = dim;

	da_scrivere.foto = new Mat[dim];
	da_scrivere.indirizzo = new string[dim];

	/*se il resto è ZERO significa che la larghezza oppure altezza dell'immagine
	è un multiplo della larghezza predisposta della tail

	altrimenti

	SE NON E' NULLO significa che rimane fuori qualche pixel
	il resto indica quanti ne rimangono fuori*/



	int x = 0; //coordinata della colonna
	int y = 0; //coordinata della riga
	int k = 0; //contatore del vettore di path
	Rect roi1;

	//--------------------------------------------------------------------------------------
	//					ritarglio l'immagine
	//--------------------------------------------------------------------------------------


	/*vado a prendere le tail intere che stanno dentro l'area */

	/*MODIFICO LA CONDIZIONE LIMITE FOR IN CASO ABBIA DIMENSIONI MULTIPLE DI 256*/
	if (resto_rig == 0){
		div_rig = div_rig - 1;
	}

	if (resto_col == 0){
		div_col = div_col - 1;
	}
	/*
	blocco il ciclo for con il semaforo poichè qui avviene la produzione dell'oggetto scritto dal thread.
	in questo modo si evita che un secondo richiamo di funzione vada a modificare i dati già in uso da un thread 
	*/
	mu.lock();
	for (int j = 0; j <= div_rig; j = j + 1){//righe = altezza(h) <-----> y

		y = j * h;
		//cout << "j = " << j << endl;
		if ((j == div_rig) && (resto_rig != 0)){
			//cout <<"  ultima riga"<< endl;
			/*
			eseguito solo all'ultima riga
			*/


			h = resto_rig;//modifica l'altezza dell'ultima riga da isolare
			//è possibile farlo senza remore perchè dopo non serve più
		}
		for (int i = 0; i <= div_col; i = i + 1){//colonne = larghezza(w) <-----> x

			tail = cv::Mat::zeros(tail.size(), tail.type());//svuoto la tail
			//cout << "                    i = " << i << endl;
			x = i * w;
			if ((i == div_col) && (resto_col != 0)){//controlla se ci si trova nell'ultima colonna più piccola di w
				//cout << "                             ultima col  riga =" <<j<< endl;

				roi1 = Rect(x, y, resto_col, h);
				//cout << "estrazione rettangolo";
				//cout << " x :" << x;
				//cout << " y :" << y;
				//cout << " (" << resto_col<<" . "<<h<<")"<<endl;
			}
			else{	//si necessita l'else perche w serve intatto
				//cout << "estrazione rettangolo";
				//cout << " x :" << x;
				//cout << " y :" << y;
				//cout << " (" << w << " . " << h << ")" << endl;
				roi1 = Rect(x, y, w, h);
			}
			//prendo il pezzetto di immagine che mi interessa
			tmp = img(roi1);

			//incollo il ritaglio nella tail standard 
			tmp.copyTo(tail(cv::Rect(0, 0, tmp.cols, tmp.rows)));

			//salvo l'indirizzo della memoria fisica nel vettore di stringhe
			paths[k] = root + "img_" + to_string(y) + "_" + to_string(x) + ".png";

			//cout << paths[k] << endl;

			//scrivo l'immagine in memoria fisica ------- operazione lenta
			//imwrite(paths[k], tail);

			
			//da_scrivere.foto[k] = tail.clone();

			tail.copyTo(da_scrivere.foto[k]);

			da_scrivere.indirizzo[k] = paths[k];


			//svuoto la tail
			tail = cv::Mat::zeros(tail.size(), tail.type());

			//incremento l'indice del vettore dei path
			k++;
			

			//mu.lock();//blocco tail poichè avendola passata al thread devo aspettare prima di modificarla
			
			//mu.unlock();//sblocco in modo da non creare deadlock

		}

	}
	mu.unlock();
	/*
	scrivo i fotogrammi
	*/
	thread t1(scrivi_disco,nullptr);//esegue in un nuovo thread per la scrittura dei fotogrammi
	//cout << "ID THREAD " << t1.get_id() << endl;
	t1.detach();//lascia il thread libero di proseguire autonomamente 
	
	return paths;
}


int main(int argc, char *argv[]){
	
		
	std::string str;
	string app;
	string* indirizzi_path;

	// prendo in input il path dell'immagine da elaborare da linea di comando

	if (argc != 2){ // argc deve essere pari a 2
		//argv[0] nome programma
		//argv[1] path immagine
		cout << "path immagine non inserito" << endl;
		cin.get();
		return(-1);
	}else{
		str = string(argv[1]);
	}
	
	//VARIABILI PER IL CALCOLO DEL TEMPO DI ESECUZIONE
	clock_t start, end;
	double tempo;
	int NUM;

	//CREO IL FILE XML SE NON ESISTE ALTRIMENTI LO APRO
	string filename = "dati.xml";

	if (!ifstream(filename)){
		//se il file non esiste lo creo - VIENE ESEGUITO SOLAMENTE SE NON ESISTE (una sola volta)
		FileStorage fs(filename, FileStorage::WRITE);
		//inserisco i dati nel file

		//fs << "path_immagine" << "C:\\Users\\fabri\\Desktop\\MaterialeTirocinio\\prova2.tif";
		
		fs << "path_tail" << "C:\\tail\\";
		fs << "dim_vet_rid" << 10;
		fs.release();
	}
	//in ogni caso il file deve essere letto quindi si necessita di un IF a parte

	if (ifstream(filename)){
		//se il file esiste lo apro in lettura
		cout << "il file di input esiste" << endl;

		//apro il file in lettura
		FileStorage fs(filename, FileStorage::READ);

		//leggo i dati
		
		//str = (string)fs["path_immagine"];//path immagine da elaborare
		
		root = (string)fs["path_tail"];//path cartella contenente tail elaborate
		NUM = (int)fs["dim_vet_rid"];//NUMERO DI FATTORI DI RIDUZIONE
		app = root;
		fs.release();
	}
	
	//alloco il vettore contenente i fattori di riduzione
	float* array_reduction = new float[NUM];
	
	array_reduction[0] = 1.0;
	for (int i = 1; i < NUM; i++){
		//il fattore di riduzione è dato da 1/(2^i)
		array_reduction[i] = (float)(1 / pow(2, i));
		//array_reduction[i] = array_reduction[i - 1] + 0.1;
	}
	// ho riempito il vettore dei fattori di riduzione
	int DIM_TILES = 256;

	for (int t = 0; t <= 2; t++){//TIPO DI INTERPOLAZIONE DEI PIXEL

		
		for (int k = 0; k < NUM; k++){//FATTORE DI SCALA

			root = app;
			//INIZIALIZZO IL CALCOLO DEL TEMPO DI ESECUZIONE
			start = clock();
			
			
			//Mat img = ridimensionaImg(str, 0.0625, 0, false);

			Mat img = ridimensionaImg(str, array_reduction[k], t, true);
			
			//Mat img = ridimensionaImg(str, 1.6, t,true);
			
			//cout << "IMMAGINE RIDIMENSIONATA" << endl;

			//ritaglio l'immagine
			indirizzi_path = tailer(img, DIM_TILES, DIM_TILES);
			//cout << "IMMAGINE RITAGLIATA" << endl;


			//FINE CALCOLO DEL TEMPO
			end = clock();
			tempo = ((double)(end - start)) / CLOCKS_PER_SEC;

			//STAMPO IL TEMPO

			//cout << "---- ";
			//cout<< "FATTORE RIDUZIONE : " << array_reduction[k];
			//cout << "    TEMPO IMPIEGATO : " << tempo;
			cout << tempo;
			//cout << "     DIMENSIONE TILES : " << DIM_TILES << "x" << DIM_TILES;
			//cout << "     TIPO INTERPOLAZIONE : " << t;
			cout << endl;


		}
		cout << "FINE TIPOLOGIA INTERPOLAZIONE    "  << "    PREMI UN TASTO PER CONTINUARE" << endl;
		cin.get();
	}
	cout <<endl<< "FINE ESECUZIONE" << endl;
	cin.get();
	//system("pause");
	//return 0;

}