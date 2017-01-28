# MPIMatrixOperations

----------

**Code Explanation**

>- **Creating and distributing Random numbers**
Matrix rows are given using -n in the code at the end of the report. Generation ofnumbers in matrix is done using random function. The numbers are generated at one node and then distributed to all others.<br/>
Function: float * meshGenerate (MPI_Comm *comm_new, int local_rank, int num_procs, char *proc_name, int *elem_node);

>- **Topology Creation 2D Mesh.**
Creating 2d-partitioning on mesh topology:
The following function generates the mesh topology:<br/>
Void generate2DTolopy (MPI_Comm *comm_new, int *local_rank, int *num_procs)
float *meshGenerate(MPI_Comm *comm_new, int local_rank, int num_procs, char *proc_name, int *elem_per_node);

>- **Reason for 2D Mesh**
Matrix is divided between the processes and each node processes the data andsends to the right and bottom nodes.
Initially create the topology:<br/>
MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, 0, comm_new);
Parameter Ranges: n {2520, 5040, 10080}, total nodes {1, 4, 9}, p {1, 3, 4}, k {1, 3, 4}.

>- **Passing variables to the input function**
parse_arguments(int argc, char *argv); gets the arguments such as n, p, k, etc.<br/>
Command line input is needed to pass the array size and the number of nodes. Also,
the inputs are required as integer so the atoi() function is used to convert the input
