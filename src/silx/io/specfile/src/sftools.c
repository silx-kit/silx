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
 *   File:          sftools.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   General library tools
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2004/05/12 16:57:02 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sftools.c,v $
 *   Log: Revision 1.2  2004/05/12 16:57:02  sole
 *   Log: Windows support
 *   Log:
 *   Log: Revision 1.1  2003/09/12 10:34:11  sole
 *   Log: Initial revision
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 *   Log: Revision 2.2  2000/12/20 12:12:08  rey
 *   Log: bug corrected with SfAllMotors
 *   Log:
 * Revision 2.1  2000/07/31  19:05:07  19:05:07  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
#include <SpecFile.h>
#include <SpecFileP.h>

#ifdef WIN32
#include <stdio.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

/*
 * Library Functions
 */
DllExport void        freePtr         ( void *ptr );
DllExport void        freeArrNZ       ( void ***ptr, long lines );
DllExport void        SfShow          (SpecFile *sf);
DllExport void        SfShowScan      (SpecFile *sf, long index);

/*
 * Function declaration
 */
void        freeArr         ( void ***ptr, long lines );

int         sfSetCurrent    ( SpecFile *sf,   long index, int *error );
int         sfSameFile      ( SpecFile *sf,   ObjectList *list );
int         sfSameScan      ( SpecFile *sf,   long index );

int         findIndex       ( void *scan,   void *number );
int         findNoAndOr     ( void *scan,   void *number );
int         findFirst       ( void *scan,   void *file_offset );
ObjectList *findScanByIndex ( ListHeader *list, long index );
ObjectList *findFirstInFile ( ListHeader *list, long file_offset );
ObjectList *findScanByNo    ( ListHeader *list, long scan_no, long order );

long        mulstrtod       ( char *str,        double **arr, int *error );
void        freeAllData     ( SpecFile *sf );

/*
 * Globals
 */


/*********************************************************************
 *   Function:		void sfSetCurrent( sf, list )
 *
 *   Description:	Sets 'list' to current scan.
 *			Updates SpecFile structure.
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) New scan
 *
 *********************************************************************/
int
sfSetCurrent( SpecFile *sf, long index,int *error )
{
     ObjectList *list,
                *flist;
     SpecScan   *scan,
                *fscan;
     long        nbytes;
     long        fileheadsize,start;

    /*
     * If same scan nothing to do
     */
     if (sfSameScan(sf,index)) return(0);

    /*
     * It is a new scan. Free memory allocated for previous one.
     */
     freeAllData(sf);

    /*
     * Find scan
     */
     list = findScanByIndex(&(sf->list),index);

     if (list == (ObjectList *)NULL) {
         *error = SF_ERR_SCAN_NOT_FOUND;
         return(-1);
     }

    /*
     * Read full scan into buffer
     */
     scan = list->contents;

     if (sf->scanbuffer != ( char * ) NULL) free(sf->scanbuffer);

     sf->scanbuffer = ( char *) malloc(scan->size);

     if (sf->scanbuffer == (char *)NULL) {
         *error = SF_ERR_MEMORY_ALLOC;
         return(-1);
     }

     lseek(sf->fd,scan->offset,SEEK_SET);
     nbytes = read(sf->fd,sf->scanbuffer,scan->size);
     if ( nbytes == -1) {
         *error = SF_ERR_FILE_READ;
         return(-1);
     }
     if ( sf->scanbuffer[0] != '#' || sf->scanbuffer[1] != 'S') {
         *error = SF_ERR_FILE_READ;
         return(-1);
     }
     sf->scanheadersize = scan->data_offset - scan->offset;

    /*
     * if different file read fileheader also
     */
     if (!sfSameFile(sf,list)) {
        if (sf->filebuffer != ( char * ) NULL) free(sf->filebuffer);

        start        = scan->file_header;
        flist        = findFirstInFile(&(sf->list),scan->file_header);
        if (flist == (ObjectList *) NULL) {
            fileheadsize = 0;
            sf->filebuffersize = fileheadsize;
        }
        else
        {
            fscan        = flist->contents;
            fileheadsize = fscan->offset - start;
        }

        if (fileheadsize > 0) {
            sf->filebuffer = ( char *) malloc(fileheadsize);
            if (sf->filebuffer == (char *)NULL) {
               *error = SF_ERR_MEMORY_ALLOC;
                return(-1);
            }
            lseek(sf->fd,start,SEEK_SET);
            nbytes = read(sf->fd,sf->filebuffer,fileheadsize);
            if ( nbytes == -1) {
               *error = SF_ERR_FILE_READ;
               return(-1);
            }
            sf->filebuffersize = fileheadsize;
        }
     }
     sf->scansize = scan->size;
     sf->current  = list;

     return(1);
}


/*********************************************************************
 *   Function:		int sfSameFile( sf, list )
 *
 *   Description:	Checks if the current scan file header and
 *			the new scan file header are the same.
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) New scan
 *   Returns:
 *		1 - the same
 *		0 - not the same
 *
 *********************************************************************/
int
sfSameFile( SpecFile *sf, ObjectList *list )
{
     if (sf->current) {
     return ( ((SpecScan *)sf->current->contents)->file_header ==
	      ((SpecScan *)list->contents)->file_header  );
     } else return(0);
}


/*********************************************************************
 *   Function:		int sfSameScan( sf, index )
 *
 *   Description:	Checks if the current scan and
 *			the new scan are the same.
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) New scan index
 *   Returns:
 *		1 - the same
 *		0 - not the same
 *
 *********************************************************************/
int
sfSameScan( SpecFile *sf, long index )
{
     if ( sf->current == (ObjectList *)NULL) return(0);

     return ( ((SpecScan *)sf->current->contents)->index == index );
}


/*********************************************************************
 *   Function:		freePtr( ptr );
 *
 *   Description:	Frees memory pointed to by 'ptr'.
 *
 *   Parameters:
 *		Input :	(1) Pointer
 *
 *********************************************************************/
void
freePtr( void *ptr )
{
     free( ptr );
}


/*********************************************************************
 *   Function:		freeArrNZ( ptr, lines );
 *
 *   Description:	Frees an array if 'lines' > zero.
 *
 *   Parameters:
 *		Input :	(1) Array pointer
 *			(2) No. of lines
 *
 *********************************************************************/
void
freeArrNZ( void ***ptr, long lines )
{
     if ( *ptr != (void **)NULL  &&  lines > 0 ) {
	  for ( ; lines ; lines-- ) {
	       free( (*ptr)[lines-1] );
	  }
	  free( *ptr );
	  *ptr = ( void **)NULL ;
     }
}


/*********************************************************************
 *   Function:		freeArr( ptr, lines );
 *
 *   Description:	Frees an array.
 *			 'ptr' will be always freed !!!
 *
 *   Parameters:
 *		Input :	(1) Array pointer
 *			(2) No. of lines
 *
 *********************************************************************/
void
freeArr( void ***ptr, long lines )
{
     if ( *ptr != (void **)NULL ) {
	  if ( lines > 0 ) {
	       for ( ; lines ; lines-- ) {
		    free( (*ptr)[lines-1] );
	       }
	  }
	  free( *ptr );
	  *ptr = ( void **)NULL ;
     }
}


/*********************************************************************
 *   Function:		int findIndex( scan, number )
 *
 *   Description:	Compares if number == scan index .
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) number
 *   Returns:
 *			0 : not found
 *			1 : found
 *
 *********************************************************************/
int
findIndex( void *scan, void *number )
{
     return( ((SpecScan *)scan)->index == *(long *)number );
}


/*********************************************************************
 *   Function:		int findFirst( scan, file_offset )
 *
 *   Description:	Compares if scan offset > file_offset
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) number
 *   Returns:
 *			0 : not found
 *			1 : found
 *
 *********************************************************************/
int
findFirst( void *scan, void *file_offset )
{
     return( ((SpecScan *)scan)->offset > *(long *)file_offset );
}


/*********************************************************************
 *   Function:		int findNoAndOr( scan, number )
 *			      ( Number
 *				     Order )
 *
 *   Description:	Compares if number1 = scan number and
 *				    number2 = scan order
 *   Parameters:
 *		Input:	(1) SpecScan pointer
 *			(2) number[1]
 *   Returns:
 *			0 : not found
 *			1 : found
 *
 *********************************************************************/
int
findNoAndOr( void *scan, void *number )
{

     long *n = (long *)number;

     return( ( ((SpecScan *)scan)->scan_no == *n++ ) && ( ((SpecScan *)scan)->order   == *n ));
}


/*********************************************************************
 *   Function:		ObjectList *findScanByIndex( list, index )
 *
 *   Description:	Looks for a scan .
 *
 *   Parameters:
 *		Input:	(1) List pointer
 *			(2) scan index
 *   Returns:
 *			ObjectList pointer if found ,
 *			NULL if not.
 *
 *********************************************************************/
ObjectList *
findScanByIndex( ListHeader *list, long index )
{
     return findInList( list, findIndex, (void *)&index );
}


/*********************************************************************
 *   Function:		ObjectList findScanByNo( list, scan_no, order )
 *
 *   Description:	Looks for a scan .
 *
 *   Parameters:
 *		Input:	(1) List pointer
 *			(2) scan number
 *			(3) scan order
 *   Returns:
 *			ObjectList pointer if found ,
 *			NULL if not.
 *
 *********************************************************************/
ObjectList *
findScanByNo( ListHeader *list, long scan_no, long order )
{
     long	 value[2];

     value[0] = scan_no;
     value[1] = order;

     return( findInList( (void *)list, findNoAndOr, (void *)value) );
}



/*********************************************************************
 *   Function:		ObjectList *findFirstInFile( list, file_offset )
 *
 *   Description:	Looks for a scan .
 *
 *   Parameters:
 *		Input:	(1) List pointer
 *			(2) scan index
 *   Returns:
 *			ObjectList pointer if found ,
 *			NULL if not.
 *
 *********************************************************************/
ObjectList *
findFirstInFile( ListHeader *list, long file_offset )
{
     return findInList( list, findFirst, (void *)&file_offset );
}


/*********************************************************************
 *   Function:        long mulstrtod( str, arr, error )
 *
 *   Description:     Converts string to data array.( double array )
 *
 *   Parameters:
 *        Input :    (1) String
 *
 *        Output:
 *            (2) Data array
 *            (3) error number
 *   Returns:
 *            Number of values.
 *            ( -1 ) in case of errors.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
long
mulstrtod( char *str, double **arr, int *error )
{
     int      count,q,i=0;
     double  *ret;
     char    *str2;
     static   double tmpret[200];

     *arr = (double *)NULL;

     str2 = str;

     while( (q = sscanf(str2, "%lf%n", &(tmpret[i]), &count)) > 0 ) {
        i++;
        str2 += count;
     }
     str2++;

     if ( !i ) {
        return( i );
     }

     ret = (double *)malloc( sizeof(double) * i );

     if ( ret == (double *)NULL ) {
        *error = SF_ERR_MEMORY_ALLOC;
         return( -1 );
     }
     memcpy(ret, tmpret, i * sizeof(double) );

     *arr = ret;
     return( i );
}

void
freeAllData(SpecFile *sf)
{
    if (sf->motor_pos != (double *)NULL) {
         free(sf->motor_pos);
         sf->motor_pos    = (double *)NULL;
         sf->no_motor_pos = -1;
    }
    if (sf->motor_names != (char **)NULL) {
         freeArrNZ((void ***)&(sf->motor_names),sf->no_motor_names);
         sf->motor_names    = (char **)NULL;
         sf->no_motor_names = -1;
    }
    if (sf->labels != (char **)NULL) {
         freeArrNZ((void ***)&(sf->labels),sf->no_labels);
         sf->labels    = (char **)NULL;
         sf->no_labels = -1;
    }
    if (sf->data_info != (long *)NULL) {
         freeArrNZ((void ***)&(sf->data),sf->data_info[ROW]);
         free(sf->data_info);
         sf->data      = (double **)NULL;
         sf->data_info = (long *)NULL;
    }
}

DllExport void
SfShow          (SpecFile *sf) {
      printf("<Showing Info>  - specfile: %s\n",sf->sfname);
      printf("    - no_scans: %ld\n",sf->no_scans);
      printf("    - current:  %ld\n",((SpecScan*)sf->current->contents)->scan_no);
      printf("    Cursor:\n");
      printf("    - no_scans: %ld\n",sf->cursor.scanno);
      printf("    - bytecnt:  %ld\n",sf->cursor.bytecnt);
}

DllExport void
SfShowScan      (SpecFile *sf, long index) {
     int       error;
     SpecScan *scan;

     printf("<Showing Info>  - specfile: %s / idx %ld\n",sf->sfname,index);

     if (sfSetCurrent(sf,index,&error) == -1) {
         printf("Cannot get scan index %ld\n",index);
     }

     scan = (SpecScan *) sf->current->contents;

     printf("     - index:         %ld\n",scan->index);
     printf("     - scan_no:       %ld\n",scan->scan_no);
     printf("     - offset:        %ld\n",scan->offset);
     printf("     - data_offset:   %ld\n",scan->data_offset);
}
