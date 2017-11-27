//cArray.c//

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    double **c, *a;
    int i,n,m;
    
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i]=a+i*m;  }
    return c;
}


/*Function that returns a random number between a range, C internal use*/
double randomInRange(double min, double max)
{
    double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */

}


/* ==== Operate on Matrix components  =========================
   Access to the introduced matrix data to calculate 3d points using MonteCarlo funciont with random numbers
  
    Sets the values in the inserted matout NumPy array and gives 3d space points
    interface:  matrix2D(matin, matout)
                matin is the psi values NumPy matrix, matout is the output values matrix(matrix[number_points][3] )
*/
static PyObject* matrix2D(PyObject* self, PyObject* args)
{
    PyArrayObject *matin, *matout;
    double **cin, **cout;   
    int n = 0;                      //If 0 at debug then something failed, else returns 1
    int f_in, c_in, f_out, c_out;

    if (!PyArg_ParseTuple(args, "OO", &matin, &matout))
        return NULL;

    cin=pymatrix_to_Carrayptrs(matin);
    cout=pymatrix_to_Carrayptrs(matout);

    
    f_in=matin->dimensions[0];
    c_in=matin->dimensions[1];

    f_out=matout->dimensions[0];
    c_out=matout->dimensions[1];

    double random = 0;
    int random_pointer_x = 0;
    int random_pointer_y = 0;

    for (int i=0; i<f_out; i++)  {
        random = randomInRange(0,1.0);
        random_pointer_x = (int) randomInRange(0,f_in);
        random_pointer_y = (int) randomInRange(0,c_in);
        while (random > cin[random_pointer_x][random_pointer_y])
        {
            random = randomInRange(0,1);
            random_pointer_x = (int) randomInRange(0,f_in);
            random_pointer_y = (int) randomInRange(0,c_in);
        }
        cout[i][0] = random_pointer_x - f_in/2 + randomInRange(0,0.82); //n has the Z size (len(Z[n](m))), with this substraction operation valuer are better balanced
        cout[i][1] = random_pointer_y - f_in/2 + randomInRange(0,0.82); //because the final objective is show them in a 3D grid
        cout[i][2] = random;
    }

    n=1;
    return Py_BuildValue("i", n);
}

/* ==== Operate on Matrix components  =========================
   Access to the introduced matrix data to calculate 3d points using MonteCarlo funciont with random numbers
  
    Sets the values in the inserted matout NumPy array and gives 3d space points and the probability [4]
    interface:  matrix2Dprob(matin, matout)
                matin is the psi values NumPy matrix, matout is the output values matrix(matrix[number_points][4] )
*/
static PyObject* matrix2Dprob(PyObject* self, PyObject* args)
{
    PyArrayObject *matin, *matout;
    double **cin, **cout;   
    int n = 0;                      //If 0 at debug then something failed, else returns 1
    int f_in, c_in, f_out, c_out;

    if (!PyArg_ParseTuple(args, "OO", &matin, &matout))
        return NULL;

    cin=pymatrix_to_Carrayptrs(matin);
    cout=pymatrix_to_Carrayptrs(matout);

    
    f_in=matin->dimensions[0];
    c_in=matin->dimensions[1];

    f_out=matout->dimensions[0];
    c_out=matout->dimensions[1];

    double random = 0;
    int random_pointer_x = 0;
    int random_pointer_y = 0;

    for (int i=0; i<f_out; i++)  {
        random = randomInRange(0,1.0);
        random_pointer_x = (int) randomInRange(0,f_in);
        random_pointer_y = (int) randomInRange(0,c_in);
        while (random > cin[random_pointer_x][random_pointer_y])
        {
            random = randomInRange(0,1);
            random_pointer_x = (int) randomInRange(0,f_in);
            random_pointer_y = (int) randomInRange(0,c_in);
        }
        cout[i][0] = random_pointer_x - f_in/2 + randomInRange(0,1); //n has the Z size (len(Z[n](m))), with this substraction operation valuer are better balanced
        cout[i][1] = random_pointer_y - f_in/2 + randomInRange(0,1); //because the final objective is show them in a 3D grid
        cout[i][2] = random;
        cout[i][3] = cin[random_pointer_x][random_pointer_y];
    }

    n=1;
    return Py_BuildValue("i", n);
}



/* ==== Operate on Matrix components  =========================
   Access to the introduced matrix data to calculate 3d points using MonteCarlo funciont with random numbers
   matin now is a 3D array like this matin[x][y][z] with the psi values
    Sets the values in the inserted matout NumPy array and gives 3d space points and the probability [4]
    interface:  matrix3Dprob(matin, matout)
                matin is the psi values NumPy matrix, matout is the output values matrix(matrix[number_points][4] )
*/
static PyObject* matrix3Dprob(PyObject* self, PyObject* args)
{
    PyArrayObject *matin, *matout;
    double **cout;   
    int n = 0;                      //If 0 at debug then something failed, else returns 1
    int f_in, c_in, e_in, f_out, c_out;

    if (!PyArg_ParseTuple(args, "OO", &matin, &matout))
        return NULL;

    cout=pymatrix_to_Carrayptrs(matout);

    f_in=matin->dimensions[0];
    c_in=matin->dimensions[1];
    e_in=matin->dimensions[2];

    f_out=matout->dimensions[0];
    c_out=matout->dimensions[1];

    double random = 0;
    int rand_x = 0;
    int rand_y = 0;
    int rand_z = 0;

    for (int i=0; i<f_out; i++)  {
        random = randomInRange(0,1.0);
        rand_x = (int) randomInRange(0,f_in);
        rand_y = (int) randomInRange(0,c_in);
        rand_z = (int) randomInRange(0,e_in);
        while (random > *((double *)PyArray_GETPTR3(matin,rand_x,rand_y,rand_z)))
        {
            random = randomInRange(0,1);
            rand_x = (int) randomInRange(0,f_in);
            rand_y = (int) randomInRange(0,c_in);
            rand_z = (int) randomInRange(0,e_in);
        }
        cout[i][0] = rand_x - f_in/2 + randomInRange(0,1); //with this substraction operation values are better balanced
        cout[i][1] = rand_y - c_in/2 + randomInRange(0,1); //because the final objective is show them in a 3D grid
        cout[i][2] = rand_z - e_in/2 + randomInRange(0,1); //now this value is real too, because comes from 3D data
        cout[i][3] = *((double *)PyArray_GETPTR3(matin,rand_x,rand_y,rand_z));
    }

    n=1;
    return Py_BuildValue("i", n);
}

/*Function that returns a random number between a range*/
static PyObject* randInRangeD(PyObject* self, PyObject* args)
{
    double min, max, n;

    if (!PyArg_ParseTuple(args, "dd", &min, &max))
        return NULL;

    double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
    n = min + scale * ( max - min );      /* [min, max] */

    return Py_BuildValue("d", n);
}
 
double Cfib(double n)
{
    if (n < 2)
        return n;
    else
        return Cfib(n-1) + Cfib(n-2);
}
 
static PyObject* fib(PyObject* self, PyObject* args)
{
    double n;
    double m; 

    if (!PyArg_ParseTuple(args, "dd", &n, &m))
        return NULL;

    return Py_BuildValue("d", Cfib(n*m));
}

static PyObject* version(PyObject* self)
{
    return Py_BuildValue("s", "Version 1.x4");
}
 
static PyMethodDef myMethods[] = {
    {"randInRangeD", randInRangeD, METH_VARARGS, "Returns a float number between two values randInRange(min, max) min included"},
    {"matrix2D", matrix2D, METH_VARARGS, "Calculates the MonteCarlo with the imput array and saves results in the other array (ArrayImput, ArrayOutput) ArrayOutput[any_size][3]"},
    {"matrix2Dprob", matrix2Dprob, METH_VARARGS, "Calculates the MonteCarlo with the imput array and saves results in the other array (ArrayImput, ArrayOutput) returns probability too ArrayOutput[any_size][4]"},
    {"matrix3Dprob", matrix3Dprob, METH_VARARGS, "Calculates the MonteCarlo with the imput array (3d) and saves results in the other array (ArrayImput, ArrayOutput) returns probability too ArrayOutput[any_size][4]"},
    {"fib", fib, METH_VARARGS, "Calculate the Fibonacci numbers."},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef cArray = {
	PyModuleDef_HEAD_INIT,
	"cArray", //name of module.
	"Array calculation Module",
	-1,
	myMethods
};

PyMODINIT_FUNC PyInit_cArray(void)
{
    return PyModule_Create(&cArray);
    import_array();
}

