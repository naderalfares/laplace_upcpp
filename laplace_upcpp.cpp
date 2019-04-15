/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  John Urbanic, PSC 2014
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <upcxx/upcxx.hpp>




// size of plate
#define COLUMNS    1000
#define ROWS       1000

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

/*
double Temperature[ROWS+2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS+2][COLUMNS+2]; // temperature grid from last iteration
*/

//   helper routines
void initialize(double * local_temp);
void track_progress(int iter);


int main(int argc, char *argv[]) {


    upcxx::init();
    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
    struct timeval start_time, stop_time, elapsed_time;  // timers
    


    gettimeofday(&start_time,NULL); // Unix timer
    
    max_iterations = atoi(argv[1]);

    
    int rank;
    int num_of_procs = upcxx::rank_n();

    rank = upcxx::rank_me();
    
    std::cout<< "iterations: " << max_iterations <<  " ranks: " << num_of_procs << " my_rank: " << rank << std::endl;
    
    int total_size = (COLUMNS+2) * (ROWS+2);
    int size_per_proc = total_size / num_of_procs;
    
    upcxx::dist_object<upcxx::global_ptr<double>>  Temprature_last (upcxx::new_array<double>(total_size));
    double * local_temp = Temprature_last->local();

    initialize(local_temp);
    upcxx::barrier();

    upcxx::global_ptr<double> temp_rank_prev = nullptr, temp_rank_next = nullptr;
    
    double Temperature_new [ROWS+2][COLUMNS+2];

    
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
        printf("iteration %d\n", iteration);
        double *temp; 
        int first_row_index = ((ROWS+2)/num_of_procs)*upcxx::rank_me();
        int last_row_index = (( (ROWS+2)/num_of_procs)*upcxx::rank_me()+1) -1;
        // main calculation: average my four neighbors
        for(i = first_row_index+1; i <= last_row_index-1 ; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                //temp = upcxx::rget(local_temp).wait();
                temp = local_temp;
                Temperature_new[i][j] = 0.25 * (temp[(i + 1) * COLUMNS + j] + temp[(i-1) * COLUMNS + j]+
                                                temp[i * COLUMNS + j + 1] + temp[i * COLUMNS + j - 1]);
            }
        }

        if( rank == 0 ){
            temp_rank_next = Temprature_last.fetch(upcxx::rank_me() + 1).wait();
            //temp = upcxx::rget(temp_rank_next).wait();
            temp = temp_rank_next.local();
            i = last_row_index; 
            for(j = 1; j <= COLUMNS; j++) 
                Temperature_new[i][j] = 0.25 * (temp[(i + 1) * COLUMNS + j] + temp[(i-1) * COLUMNS + j]+
                                                temp[i * COLUMNS + j + 1] + temp[i * COLUMNS + j - 1]);
                       
        }else if( rank == upcxx::rank_n() - 1){
            temp_rank_prev = Temprature_last.fetch(upcxx::rank_me() - 1).wait();
            //temp = upcxx::rget(temp_rank_prev).wait();
            temp = temp_rank_prev.local();
            i = first_row_index; 
            for(j = 1; j <= COLUMNS; j++) 
                Temperature_new[i][j] = 0.25 * (temp[(i + 1) * COLUMNS + j] + temp[(i-1) * COLUMNS + j]+
                                                temp[i * COLUMNS + j + 1] + temp[i * COLUMNS + j - 1]);
            


        }else{
            temp_rank_prev = Temprature_last.fetch(upcxx::rank_me() - 1).wait();
            temp_rank_next = Temprature_last.fetch(upcxx::rank_me() + 1).wait();
            //temp = upcxx::rget(temp_rank_prev).wait();
            temp = temp_rank_prev.local();
            
            i = first_row_index; 
            for(j = 1; j <= COLUMNS; j++) 
                Temperature_new[i][j] = 0.25 * (temp[(i + 1) * COLUMNS + j] + temp[(i-1) * COLUMNS + j]+
                                                temp[i * COLUMNS + j + 1] + temp[i * COLUMNS + j - 1]);
            

            //temp = upcxx::rget(temp_rank_next).wait();
            temp = temp_rank_next.local();
            i = last_row_index; 
            for(j = 1; j <= COLUMNS; j++) 
                Temperature_new[i][j] = 0.25 * (temp[(i + 1) * COLUMNS + j] + temp[(i-1) * COLUMNS + j]+
                                                temp[i * COLUMNS + j + 1] + temp[i * COLUMNS + j - 1]);
        }
        

        //temp = upcxx::rget(local_temp).wait();
        temp = local_temp;
        // main calculation: average my four neighbors
        for(i = first_row_index; i <= last_row_index ; i++) {
            for(j = 1; j <= COLUMNS; j++) {
               if(i==1 || i==ROWS) continue;
               dt = fmax( fabs(Temperature_new[i][j] - temp[i*COLUMNS + j]), dt);
               temp[i * COLUMNS + j] = Temperature_new[i][j];
            }
        }


        upcxx::barrier();
        double max_dt = upcxx::reduce_all(dt, upcxx::op_fast_max).wait();
        dt = max_dt;
        
        iteration++;
        
    }



    if(rank == 0){
         gettimeofday(&stop_time,NULL);
         timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
         printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
         printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    }
         


    /*     
         printf("Maximum iterations [100-4000]?\n");
         scanf("%d", &max_iterations);

         gettimeofday(&start_time,NULL); // Unix timer

        
         initialize();                   // initialize Temp_last including boundary conditions

         // do until error is minimal or until max steps
         while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

             // main calculation: average my four neighbors
             for(i = 1; i <= ROWS; i++) {
                 for(j = 1; j <= COLUMNS; j++) {
                     Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                                 Temperature_last[i][j+1] + Temperature_last[i][j-1]);
                 }
             }
             
             dt = 0.0; // reset largest temperature change

             // copy grid to old grid for next iteration and find latest dt
             for(i = 1; i <= ROWS; i++){
                 for(j = 1; j <= COLUMNS; j++){
               dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
               Temperature_last[i][j] = Temperature[i][j];
                 }
             }

             // periodically print test values
             if((iteration % 100) == 0) {
             track_progress(iteration);
             }

         iteration++;
         }

         gettimeofday(&stop_time,NULL);
         timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine



         printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
         printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    }
    */
    upcxx::finalize();



}




/* OLD CODE FOR INIT
// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(){

    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }
}
*/


void initialize(double *local_temp){
    int i,j;
    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            local_temp[i * COLUMNS + j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        local_temp[i*COLUMNS] = 0.0;
        local_temp[i*COLUMNS+ COLUMNS + 1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        local_temp[j] = 0.0;
        local_temp[(ROWS+1) * COLUMNS + j] = (100.0/COLUMNS)*j;
    }

}


/*
// print diagonal in bottom right corner where most action is
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = ROWS-5; i <= ROWS; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}
*/
