# /*##########################################################################
# Copyright (C) 1995-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
/************************************************************************
 *
 *   File:          sfdata.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Functions for getting data
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2005/07/04 15:02:38 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sfdata.c,v $
 *   Log: Revision 1.8  2005/07/04 15:02:38  ahoms
 *   Log: Fixed memory leak in SfNoDataLines
 *   Log:
 *   Log: Revision 1.7  2004/01/20 09:23:50  sole
 *   Log: Small change in sfdata (ptr < (to-1)) changed to (ptr <= (to-1))
 *   Log:
 *   Log: Revision 1.6  2003/03/06 16:56:40  sole
 *   Log: Check if to is beyond the scan size in SfData (still not finished but it seems to solve a crash)
 *   Log:
 *   Log: Revision 1.5  2002/12/09 13:04:05  sole
 *   Log: Added a check in SfNoDataLines
 *   Log:
 *   Log: Revision 1.4  2002/11/13 15:02:38  sole
 *   Log: Removed some printing in SfData
 *   Log:
 *   Log: Revision 1.3  2002/11/12 16:22:07  sole
 *   Log: WARNING: Developing version - Improved MCA reading and reading properly the end of the file.
 *   Log:
 *   Log: Revision 1.2  2002/11/12 13:15:52  sole
 *   Log: 1st version from Armando. The idea behind is to take the last line only if it ends with \n
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 * Revision 2.1  2000/07/31  19:05:11  19:05:11  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 *
 */
#include <SpecFile.h>
#include <SpecFileP.h>
#include <locale_management.h>

#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
#include <locale.h>
#endif
#endif

#include <ctype.h>
/*
 * Define macro
 */
#define isnumber(this) ( isdigit(this) || this == '-' || this == '+' || this == '.' || this == 'E' || this == 'e')

/*
 * Mca continuation character
 */
#define MCA_CONT '\\'
#define D_INFO   3

/*
 * Declarations
 */
DllExport long SfNoDataLines  ( SpecFile *sf, long index, int *error );
DllExport int  SfData         ( SpecFile *sf, long index, double ***retdata,
                                          long **retinfo, int *error );
DllExport long SfDataAsString ( SpecFile *sf, long index,
                                          char ***data, int *error );
DllExport long SfDataLine     ( SpecFile *sf, long index, long line,
                                          double **data_line, int *error );
DllExport long SfDataCol      ( SpecFile *sf, long index, long col,
                                          double **data_col, int *error );
DllExport long SfDataColByName( SpecFile *sf, long index,
                                  char *label, double **data_col, int *error );


/*********************************************************************
 *   Function:        long SfNoDataLines( sf, index, error )
 *
 *   Description:    Gets number of data lines in a scan
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Number of data lines ,
 *            ( -1 ) => errors.
 *   Possible errors:
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *********************************************************************/
DllExport long
SfNoDataLines( SpecFile *sf, long index, int *error )
{
    long     *dinfo    = NULL;
    double  **data     = NULL;
    long      nrlines  = 0;
    int       ret, i;

    ret = SfData(sf,index,&data,&dinfo,error);

    if (ret == -1) {
        return(-1);
    }
    if (dinfo == (long *) NULL){
        return(-1);
    }
    if (dinfo[ROW] < 0){
        printf("Negative number of points!\n");
        /*free(dinfo);*/
        return(-1);
    }

    nrlines = dinfo[ROW];

    /* now free all stuff that SfData allocated */
    for (i = 0; i < nrlines; i++)
        free(data[i]);
    free(data);
    free(dinfo);

    return nrlines;
}



/*********************************************************************
 *   Function:        int SfData(sf, index, data, data_info, error)
 *
 *   Description:    Gets data.
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) Data array
 *            (4) Data info : [0] => no_lines
 *                    [1] => no_columns
 *                    [2] = ( 0 ) => regular
 *                          ( 1 ) => not regular !
 *            (5) error number
 *   Returns:
 *            (  0 ) => OK
 *                ( -1 ) => errors occured
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *            SF_ERR_FILE_READ
 *            SF_ERR_SCAN_NOT_FOUND
 *            SF_ERR_LINE_NOT_FOUND
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport int
SfData( SpecFile *sf, long index, double ***retdata, long **retinfo, int *error )
{
     long     *dinfo    = NULL;
     double  **data     = NULL;
     double   *dataline = NULL;
     long      headersize;

     char *ptr,
          *from,
          *to;

     char    strval[100];
     double  val;
     double  valline[512];
     long    cols,
             maxcol=512;
     long    rows;
     int     i;
#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
	char *currentLocaleBuffer;
	char localeBuffer[21];
#endif
#endif

     if (index <= 0 ){
        return(-1);
     }

     if (sfSetCurrent(sf,index,error) == -1 )
             return(-1);


     /*
      * Copy if already there
      */
     if (sf->data_info != (long *)NULL) {
          dinfo = ( long * ) malloc ( sizeof(long) * D_INFO);
          dinfo[ROW] = sf->data_info[ROW];
          dinfo[COL] = sf->data_info[COL];
          dinfo[REG] = sf->data_info[REG];
          data =  ( double **) malloc ( sizeof(double *) * dinfo[ROW]);
          for (i=0;i<dinfo[ROW];i++) {
              data[i] = (double *)malloc (sizeof(double) * dinfo[COL]);
              memcpy(data[i],sf->data[i],sizeof(double) * dinfo[COL]);
          }
          *retdata = data;
          *retinfo = dinfo;
          return(0);
     }
     /*
      * else do the job
      */

     if ( ((SpecScan *)sf->current->contents)->data_offset == -1 ) {
          *retdata = data;
          *retinfo = dinfo;
          return(-1);
     }

     headersize = ((SpecScan *)sf->current->contents)->data_offset
                - ((SpecScan *)sf->current->contents)->offset;

     from = sf->scanbuffer + headersize;
     to   = sf->scanbuffer + ((SpecScan *)sf->current->contents)->size;
     if (to > sf->scanbuffer+sf->scansize){
          /* the -32 found "experimentaly" */
          ptr = sf->scanbuffer+sf->scansize - 32;
          while (*ptr != '\n') ptr--;
          to=ptr;
          /*printf("I let it crash ...\n");*/
     }
     i=0;
     ptr = from;
     rows = -1;
     cols = -1;
     /*
      * Alloc memory
      */
     if ( (data = (double **) malloc (sizeof(double *)) ) == (double **)NULL) {
         *error = SF_ERR_MEMORY_ALLOC;
          return(-1);
     }

     if ( (dinfo = (long *) malloc(sizeof(long) * D_INFO) ) == (long *)NULL) {
          free(data);
         *error = SF_ERR_MEMORY_ALLOC;
          return(-1);
     }
     ptr = from;
     dinfo[ROW] = dinfo[COL] = dinfo[REG] = 0;

#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
	currentLocaleBuffer = setlocale(LC_NUMERIC, NULL);
	strcpy(localeBuffer, currentLocaleBuffer);
	setlocale(LC_NUMERIC, "C\0");
#endif
#endif
    for ( ; ptr < to; ptr++) {
        /* get a complete line */
        i=0;
        cols=0;
        /*I should be at the start of a line */
        while(*(ptr) != '\n'){
            if (*(ptr-1) == '\n'){
                /*I am at the start of a line */
                while(*ptr == '#'){
                    if (ptr >= to)
                        break;
                    for (; ptr < to; ptr++){
                        if (*ptr == '\n'){
                            break;
                        }
                    };
                    /* on exit is equal to newline */
                    if (ptr < to) {
                        ptr++;
                    }
                }
                if (*ptr == '@') {
                     /*
                    * read all mca block: go while in buffer ( ptr < to - 1 )
                    * and while a newline is preceded by a slash
                    */
                    for (    ptr = ptr + 2;
                        (*ptr != '\n' || (*(ptr-1) == MCA_CONT)) && ptr < to ;
                        ptr++);
                    if (ptr >= to){
                        break;
                    }
                }
                while(*ptr == '#'){
                    if (ptr >= to)
                        break;
                    for (; ptr < to; ptr++){
                        if (*ptr == '\n'){
                            break;
                        }
                    };
                    /* on exit is equal to newline */
                    if (ptr < to) {
                        ptr++;
                    }
                }
                /* first characters of buffer
                */
                while (*ptr == ' ' && ptr < to) ptr++;  /* get rid of empty spaces */
            }
           /*
            * in the middle of a line
            */
            if (*ptr == ' ' || *ptr == '\t' ) {
                strval[i] = '\0';
                i = 0;
                val = PyMcaAtof(strval);
                valline[cols] = val;
                cols++;
                if (cols >= maxcol) return(-1);
                while(*(ptr+1) == ' ' || *(ptr+1) == '\t') ptr++;
            } else {
                if isnumber(*ptr){
                    strval[i] = *ptr;
                    i++;
                }
            }
            if (ptr >= (to-1)){
                break;
            }
            ptr++;
        }
        if ((*(ptr)== '\n') && (i != 0)){
                strval[i] = '\0';
                val = PyMcaAtof(strval);
                valline[cols] = val;
                cols++;
                if (cols >= maxcol) return(-1);
                /*while(*(ptr+1) == ' ' || *(ptr+1) == '\t') ptr++;*/
        }
        /*printf("%c",*ptr);*/
        /* diffract31 crash -> changed from i!=0 to i==0 */
        /*cols>0 necessary scan 59 of 31oct98 */
        if ((ptr < to) && (cols >0)) {
        rows++;
        /*cols++;*/
        if (cols >= maxcol) return(-1);
        /* printf("Adding a new row, nrows = %ld, ncols= %ld\n",rows,cols);*/
        /*printf("info col = %d cols = %d\n", dinfo[COL], cols);*/
        if (dinfo[COL] != 0 && cols != dinfo[COL]) {
                    ;
                    /*diffract31 crash -> nextline uncommented */
                    dinfo[REG] = 1;
        } else {
                    dinfo[COL] = cols;
        }
        if(dinfo[COL]==cols){
              dataline = (double *)malloc(sizeof(double) * cols);
              memcpy(dataline,valline,sizeof(double) * cols);
              data = (double **) realloc ( data, sizeof(double) * (rows+1));
              data[rows] = dataline;
              dinfo[ROW]=rows+1;
        }else{
              printf("Error on scan %d line %d\n", (int) index, (int) (rows+1));
              /* just ignore the line instead of stopping there with a
              break; */
              rows--;
        }
        }
    }

#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
    setlocale(LC_NUMERIC, localeBuffer);
#endif
#endif
    /*
    * make a copy in specfile structure
    */
    if ( dinfo[ROW] != 0 && dinfo[REG] == 0) {
        if (sf->data_info != (long *)NULL){
            printf("I should not be here!/n");
          sf->data_info[ROW] = dinfo[ROW];
          sf->data_info[COL] = dinfo[COL];
          sf->data_info[REG] = dinfo[REG];
          for (i=0;i<dinfo[ROW];i++) {
              sf->data[i]= (double *)realloc (sf->data[i],sizeof(double) * dinfo[COL]);
              if (sf->data[i] == (double *) NULL){
                printf("Realloc problem");
                return (-1);
              }
              memcpy(sf->data[i],data[i],sizeof(double) * dinfo[COL]);
          }
          *retdata = data;
          *retinfo = dinfo;
          return(0);
        }else{
            sf->data_info = ( long * ) malloc ( sizeof(long) * D_INFO);
            sf->data_info[ROW] = dinfo[ROW];
            sf->data_info[COL] = dinfo[COL];
            sf->data_info[REG] = dinfo[REG];
            sf->data =  ( double **) malloc ( sizeof(double *) * dinfo[ROW]);
            if (sf->data == (double **) NULL){
                    printf("malloc1 problem");
                    return (-1);
            }
            for (i=0;i<dinfo[ROW];i++) {
                sf->data[i] = (double *)malloc (sizeof(double) * dinfo[COL]);
                if (sf->data[i] == (double *) NULL){
                    printf("malloc2 problem");
                    return (-1);
                }
                memcpy(sf->data[i],data[i],sizeof(double) * dinfo[COL]);
            }
        }
    } else {
        if (dinfo[REG] == 0) {
            ;
            /*printf("Not Freeing data:!\n");*/
            /* I can be in the case of an mca without scan points */
            /*free(data);
            return(-1);*/
        }
    }
    *retinfo = dinfo;
    *retdata = data;
     return( 0 );
}


DllExport long
SfDataCol ( SpecFile *sf, long index, long col, double **retdata, int *error )
{
    double *datacol=NULL;

    long     *dinfo    = NULL;
    double  **data     = NULL;

    long     selection;
    int      i,ret;

    ret = SfData(sf,index,&data,&dinfo,error);

    if (ret == -1) {
        *error = SF_ERR_COL_NOT_FOUND;
        *retdata = datacol;
        return(-1);
    }

    if (col < 0) {
       selection = dinfo[COL] + col;
    } else {
       selection = col - 1;
    }
if (selection > dinfo[COL] - 1) {
selection=dinfo[COL] - 1;
}
    if ( selection < 0 || selection > dinfo[COL] - 1) {
       *error  = SF_ERR_COL_NOT_FOUND;
        if ( dinfo != (long *)NULL) {
           freeArrNZ((void ***)&data,dinfo[ROW]);
        }
        free(dinfo);
        return(-1);
    }

    datacol = (double *) malloc( sizeof(double) * dinfo[ROW]);
    if (datacol == (double *)NULL) {
       *error  = SF_ERR_MEMORY_ALLOC;
       if ( dinfo != (long *)NULL)
             freeArrNZ((void ***)&data,dinfo[ROW]);
       free(dinfo);
       return(-1);
    }

    for (i=0;i<dinfo[ROW];i++) {
       datacol[i] = data[i][selection];
    }

    ret = dinfo[ROW];

    if ( dinfo != (long *)NULL)
           freeArrNZ((void ***)&data,dinfo[ROW]);
    free(dinfo);

   *retdata = datacol;
    return(ret);
}


DllExport long
SfDataLine( SpecFile *sf, long index, long line, double **retdata, int *error )
{
    double *datarow=NULL;

    long     *dinfo    = NULL;
    double  **data     = NULL;

    long     selection;
    int      ret;

    ret = SfData(sf,index,&data,&dinfo,error);

    if (ret == -1) {
        *error = SF_ERR_LINE_NOT_FOUND;
        *retdata = datarow;
        return(-1);
    }

    if (line < 0) {
       selection = dinfo[ROW] + line;
    } else {
       selection = line - 1;
    }

    if ( selection < 0 || selection > dinfo[ROW] - 1) {
       *error  = SF_ERR_LINE_NOT_FOUND;
        if ( dinfo != (long *)NULL) {
           freeArrNZ((void ***)&data,dinfo[ROW]);
        }
        free(dinfo);
        return(-1);
    }

    datarow = (double *) malloc( sizeof(double) * dinfo[COL]);
    if (datarow == (double *)NULL) {
       *error  = SF_ERR_MEMORY_ALLOC;
       if ( dinfo != (long *)NULL)
             freeArrNZ((void ***)&data,dinfo[ROW]);
       free(dinfo);
       return(-1);
    }


    memcpy(datarow,data[selection],sizeof(double) * dinfo[COL]);

    ret = dinfo[COL];

    if ( dinfo != (long *)NULL)
           freeArrNZ((void ***)&data,dinfo[ROW]);
    free(dinfo);

   *retdata = datarow;
    return(ret);
}


DllExport long
SfDataColByName( SpecFile *sf, long index, char *label, double **retdata, int *error )
{

      double *datacol;

      long     *dinfo    = NULL;
      double  **data     = NULL;

      int      i,ret;

      char **labels = NULL;

      long  nb_lab,
            idx;

      short tofree=0;

      if ( sfSetCurrent(sf,index,error) == -1) {
        *retdata = (double *)NULL;
         return(-1);
      }

      if ( sf->no_labels != -1 ) {
         nb_lab = sf->no_labels;
         labels = sf->labels;
      } else {
         nb_lab = SfAllLabels(sf,index,&labels,error);
         tofree = 1;
      }

      if ( nb_lab == 0 || nb_lab == -1) {
            *retdata = (double *)NULL;
             return(-1);
      }

      for (idx=0;idx<nb_lab;idx++)
          if (!strcmp(label,labels[idx])) break;

      if ( idx == nb_lab ) {
          if  (tofree) freeArrNZ((void ***)&labels,nb_lab);
         *error = SF_ERR_COL_NOT_FOUND;
         *retdata = (double *)NULL;
          return(-1);
      }

      ret = SfData(sf,index,&data,&dinfo,error);

      if (ret == -1) {
         *retdata = (double *)NULL;
          return(-1);
      }

      datacol = (double *) malloc( sizeof(double) * dinfo[ROW]);
      if (datacol == (double *)NULL) {
          *error  = SF_ERR_MEMORY_ALLOC;
           if ( dinfo != (long *)NULL)
              freeArrNZ((void ***)&data,dinfo[ROW]);
           free(dinfo);
          *retdata = (double *)NULL;
           return(-1);
      }

      for (i=0;i<dinfo[ROW];i++) {
         datacol[i] = data[i][idx];
      }

      ret = dinfo[ROW];

      if ( dinfo != (long *)NULL)
           freeArrNZ((void ***)&data,dinfo[ROW]);
      free(dinfo);

     *retdata = datacol;

      return(ret);
}


DllExport long
SfDataAsString( SpecFile *sf, long index, char ***retdata, int *error )
{
     char **data=NULL;
     char   oneline[300];

     char *from,
          *to,
          *ptr,
          *dataline;

     long  headersize,rows;
     int   i;

     if (sfSetCurrent(sf,index,error) == -1 )
             return(-1);

     if ( ((SpecScan *)sf->current->contents)->data_offset == -1 ) {
          *retdata = data;
          return(-1);
     }

     data = (char **) malloc (sizeof(char *));

     headersize = ((SpecScan *)sf->current->contents)->data_offset
                - ((SpecScan *)sf->current->contents)->offset;

     from = sf->scanbuffer + headersize;
     to   = sf->scanbuffer + ((SpecScan *)sf->current->contents)->size;

     rows = -1;
     i    = 0;

    /*
     * first characters of buffer
     */

     ptr = from;

     if (isnumber(*ptr)) {
         rows++;
         oneline[i] = *ptr;
         i++;
     } else if (*ptr == '@') {
        /*
         * read all mca block: go while in buffer ( ptr < to - 1 )
         * and while a newline is preceded by a slash
         */
         for (    ptr = ptr + 2;
               (*(ptr+1) != '\n' || (*ptr == MCA_CONT)) && ptr < to - 1 ;
                  ptr++);
     }

    /*
     * continue
     */
     ptr++;

     for ( ; ptr < to - 1; ptr++) {
        /*
         * check for lines and for mca
         */
        if ( *(ptr-1) == '\n' ) {

           if ( i != 0 ) {
              oneline[i-1] = '\0';
              i = 0;

              dataline = (char *)strdup(oneline);
              data = (char **) realloc ( data, sizeof(char *) * (rows +1));
              data[rows] = dataline;
           }

           if ( *ptr == '@') {  /* Mca --> pass it all */
               for (    ptr = ptr + 2;
                     (*ptr != '\n' || (*(ptr-1) == MCA_CONT)) && ptr < to ;
                        ptr++);
           } else if ( *ptr == '#') {  /* Comment --> pass one line */
              for (ptr = ptr + 1; *ptr != '\n';ptr++);
           } else if ( isnumber(*ptr) ) {
              rows++;
              oneline[i] = *ptr;
              i++;
           }
        } else {
           if (rows == -1) continue;

           oneline[i] = *ptr;
           i++;
        }
     }

    /*
     * last line
     */

     if (rows != -1 && i) {
         oneline[i-1] = '\0';
         dataline = (char *)strdup(oneline);
         data = (char **) realloc ( data, sizeof(char *) * (rows+1));
         data[rows] = dataline;
     }

    *retdata = data;
     return(rows+1);
}
