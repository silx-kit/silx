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
 *   File:          sfindex.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   functions for scan numbering
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2004/05/12 16:56:47 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sfindex.c,v $
 *   Log: Revision 1.2  2004/05/12 16:56:47  sole
 *   Log: Support for windows
 *   Log:
 *   Log: Revision 1.1  2003/03/06 16:59:05  sole
 *   Log: Initial revision
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 * Revision 2.1  2000/07/31  19:05:15  19:05:15  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
/*
 *    File:     sfindex.c
 *
 *    Description:
 *
 *    Project:
 *
 *    Author:  Vicente Rey Bakaikoa
 *
 *    Date:    March 2000
 */
/*
 *    $Log: sfindex.c,v $
 *    Revision 1.2  2004/05/12 16:56:47  sole
 *    Support for windows
 *
 *    Revision 1.1  2003/03/06 16:59:05  sole
 *    Initial revision
 *
 *    Revision 3.0  2000/12/20 14:17:19  rey
 *    Python version available
 *
 * Revision 2.1  2000/07/31  19:05:15  19:05:15  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:26:55  13:26:55  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 *
 */

#include <SpecFile.h>
#include <SpecFileP.h>
#ifdef WIN32
#include <stdio.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif
#include <ctype.h>

#define  ON_COMMENT 0
#define  ON_ABO     1
#define  ON_RES     2
/*
 * Declarations
 */
DllExport long * SfList      ( SpecFile *sf, int *error );
DllExport long SfIndexes     ( SpecFile *sf, long number, long **idxlist );
DllExport long SfIndex       ( SpecFile *sf, long number, long order );
DllExport long SfCondList    ( SpecFile *sf, long cond,   long **scan_list,
                                      int *error );
DllExport long SfScanNo      ( SpecFile *sf );
DllExport int  SfNumberOrder ( SpecFile *sf, long index, long *number,
                                      long *order );
DllExport long SfNumber      ( SpecFile *sf, long index  );
DllExport long SfOrder       ( SpecFile *sf, long index  );

/*
 * Internal Functions
 */
static int checkAborted( SpecFile *sf, ObjectList *ptr, int *error );


/*********************************************************************
 *   Function:		long *SfList( sf, error )
 *
 *   Description:	Creates an array with all scan numbers.
 *
 *   Parameters:
 *		Input :	SpecFile pointer
 *   Returns:
 *			Array with scan numbers.
 *			NULL if errors occured.
 *   Possible errors:
 *			SF_ERR_MEMORY_ALLOC
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long *
SfList( SpecFile *sf, int *error )
{
     register	ObjectList	*ptr;
     long			*scan_list;
     long			 i = 0;

     scan_list = (long *)malloc( sizeof(long) * (sf->no_scans) );

     if ( scan_list == (long *)NULL ) {
	  *error = SF_ERR_MEMORY_ALLOC;
	  return( scan_list );
     }

     for ( ptr=sf->list.first ; ptr ; ptr=ptr->next ,i++) {
	  scan_list[i] = ( ((SpecScan *)(ptr->contents))->scan_no );
     }
     /*printf("scanlist[%li] = %li\n",i-1,scan_list[i-1]);*/
     return( scan_list );
}


/*********************************************************************
 *   Function:		long SfIndexes( sf, number , idxlist)
 *
 *   Description:	Creates an array with all indexes with the same scan
 *                      number.
 *
 *   Parameters:
 *                      Input  : SpecFile pointer
 *                               scan number
 *                      Output : array with scan indexes
 *   Returns:
 *			Number of indexes found
 *   Possible errors:
 *			None possible
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long
SfIndexes( SpecFile *sf, long number, long **idxlist )
{
     ObjectList     *ptr;
     long            i;
     long           *indexes;
     long           *arr;

     i = 0;
     indexes =  (long *)malloc(sf->no_scans * sizeof(long));

     for (ptr = sf->list.first; ptr; ptr=ptr->next ) {
           if ( number == ((SpecScan *)(ptr->contents))->scan_no) {
              indexes[i] = ((SpecScan *)(ptr->contents))->index;
              i++;
           }
     }

     if (i == 0)
        arr = (long *) NULL;
     else {
        arr = (long *)malloc(sizeof(long) * i);
        memcpy(arr,indexes,sizeof(long) * i);
     }

     *idxlist = arr;
     free(indexes);
     return( i );
}


/*********************************************************************
 *   Function:		long SfIndex( sf, number, order )
 *
 *   Description:	Gets scan index from scan number and order.
 *
 *   Parameters:
 *		Input :	(1) Scan number
 *			(2) Scan order
 *   Returns:
 *			Index number.
 *			(-1) if not found.
 *
 *********************************************************************/
DllExport long
SfIndex( SpecFile *sf, long number, long order )
{
     ObjectList		*ptr;

     ptr = findScanByNo( &(sf->list), number, order );
     if ( ptr != (ObjectList *)NULL )
        return( ((SpecScan *)(ptr->contents))->index );

     return( -1 );
}


/*********************************************************************
 *   Function:		long SfCondList( sf, cond, scan_list, error )
 *
 *   Description:	Creates an array with all scan numbers.
 *
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) Condition :	0 => not aborted scans ( NOT_ABORTED )
 *				       -1 =>     aborted scans ( ABORTED )
 *				       nn => more than 'nn' data lines
 *		Output: (3) Scan list
 *			(4) error code
 *   Returns:
 *			Number of found scans.
 *			( -1 ) if errors occured.
 *   Possible errors:
 *			SF_ERR_MEMORY_ALLOC
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long
SfCondList( SpecFile *sf, long cond, long **scan_list, int *error )
{
     register	ObjectList	*ptr;
     long			*list;
     long			 i = 0;
     int       retcheck;
     long      index;

     *scan_list = (long *)NULL;

     list = (long *)malloc( sizeof(long) * (sf->no_scans) );

     if ( list == (long *)NULL ) {
	  *error = SF_ERR_MEMORY_ALLOC;
	  return( -1 );
     }

     /*
      * Aborted scans .
      */
     if ( cond < 0 ) {    /*  aborted scans  */
	  for ( ptr=sf->list.first ; ptr ; ptr=ptr->next ) {

	       retcheck = checkAborted( sf, ptr, error );

	       if ( retcheck < 0 ) {
		    free( list );
		    return( -1 );
	       } else if ( retcheck > 0) {
	            list[i] = ( ((SpecScan *)(ptr->contents))->scan_no );
	            i++;
               }
	  }
     } else if ( cond == 0 ) {    /*   not aborted scans */
	  for ( ptr=sf->list.first ; ptr ; ptr=ptr->next ) {

	       retcheck = checkAborted( sf, ptr, error );

	       if ( retcheck < 0 ) {
		    free( list );
		    return( -1 );
	       } else if ( retcheck == 0 ) {
	            list[i] = ( ((SpecScan *)(ptr->contents))->scan_no );
	            i++;
               }
	  }
     } else {   /*  cond > 0   - more than n data_lines */
	  for ( ptr=sf->list.first ; ptr ; ptr=ptr->next ) {

	       index = ( ((SpecScan *)(ptr->contents))->index );
	       if ( SfNoDataLines(sf,index,error) <= cond ) continue;

	       list[i] = ( ((SpecScan *)(ptr->contents))->scan_no );
	       i++;
	  }
     }

     *scan_list = ( long * ) malloc ( i * sizeof(long));

     if ( *scan_list == (long *)NULL ) {
	  *error = SF_ERR_MEMORY_ALLOC;
	  return( -1 );
     }

     memcpy(*scan_list,list, i * sizeof(long));
     free(list);

     return( i );
}


/*********************************************************************
 *   Function:		long SfScanNo( sf )
 *
 *   Description:	Gets number of scans.
 *
 *   Parameters:
 *		Input :(1) SpecFile pointer
 *   Returns:
 *		Number of scans.
 *
 *********************************************************************/
DllExport long
SfScanNo( SpecFile *sf )
{
     return( sf->no_scans );
}


/*********************************************************************
 *   Function:		int SfNumberOrder( sf, index, number, order )
 *
 *   Description:	Gets scan number and order from index.
 *
 *   Parameters:
 *		Input :
 *			(1) SpecFile pointer
 *			(2) Scan index
 *		Output:
 *			(3) Scan number
 *			(4) Scan order
 *   Returns:
 *		( -1 ) => not found
 *		(  0 ) => found
 *
 *********************************************************************/
DllExport int
SfNumberOrder( SpecFile *sf, long index, long *number, long *order )
{
     register ObjectList	*list;

     *number = -1;
     *order  = -1;

     /*
      * Find scan .
      */
     list = findScanByIndex( &(sf->list), index );
     if ( list == (ObjectList *)NULL ) return( -1 );

     *number = ((SpecScan *)list->contents)->scan_no;
     *order  = ((SpecScan *)list->contents)->order;

     return( 0 );
}


/*********************************************************************
 *   Function:		long SfNumber( sf, index )
 *
 *   Description:	Gets scan number from index.
 *
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) Scan index
 *   Returns:
 *		Scan number.
 *		( -1 ) => not found
 *
 *********************************************************************/
DllExport long
SfNumber( SpecFile *sf, long index  )
{
     register ObjectList	*list;

     /*
      * Find scan .
      */
     list = findScanByIndex( &(sf->list), index );
     if ( list == (ObjectList *)NULL ) return( -1 );

     return( ((SpecScan *)list->contents)->scan_no );
}


/*********************************************************************
 *   Function:		long SfOrder( sf, index )
 *
 *   Description:	Gets scan order from index.
 *
 *   Parameters:
 *		Input :	(1) SpecFile pointer
 *			(2) Scan index
 *   Returns:
 *		Scan order.
 *		( -1 ) => not found
 *
 *********************************************************************/
DllExport long
SfOrder( SpecFile *sf, long index  )
{
     register ObjectList	*list;


     /*
      * Find scan .
      */
     list = findScanByIndex( &(sf->list), index );
     if ( list == (ObjectList *)NULL ) return( -1 );

     return( ((SpecScan *)list->contents)->order );
}

/*********************************************************************
 *   Function:		int checkAborted( sf, ptr, error )
 *
 *   Description:	Checks if scan was aborted or not .
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) Pointer to the scan
 *		Output:	(3) Error number
 *   Returns:
 *		(-1 )	: error
 *	        ( 0 )	: not aborted
 *	        ( 1 )	: aborted
 *   Possible errors:
 *			SF_ERR_MEMORY_ALLOC	| => readHeader()
 *			SF_ERR_FILE_READ
 *
 *********************************************************************/
static int
checkAborted( SpecFile *sf, ObjectList *ptr, int *error )
{
     long       nbytes;
     long       data_lines,size,from;
     SpecScan	*scan;
     char       *buffer,*cptr,next;
     int        state=ON_COMMENT;
     int        aborted=0;
     long       index;

     scan = ptr->contents;
     index = scan->index;

     data_lines = SfNoDataLines(sf,index,error);

     if ( scan->hdafter_offset == -1  && data_lines > 0) {
           return(0);
     } else if ( data_lines <= 0 ) {
        /*
         * maybe aborted on first point
         * we have to all to know ( but no data anyway )
         */
        size   = scan->size;
        from   = scan->offset;
     } else {
        size   = scan->last - scan->hdafter_offset;
        from   = scan->hdafter_offset;
     }

     lseek(sf->fd,from,SEEK_SET);
     buffer = ( char * ) malloc (size);
     nbytes = read(sf->fd,buffer,size);

     if (nbytes == -1 ) {
          *error = SF_ERR_FILE_READ;
           return(-1);
     }

     if (buffer[0] == '#' && buffer[1] == 'C') {
        state = ON_COMMENT;
     }

     for ( cptr = buffer + 1; cptr < buffer + nbytes - 1; cptr++) {
       /*
        * Comment line
        */
         if ( *cptr == '#' && *(cptr+1) == 'C' && *(cptr-1) == '\n') {
             state = ON_COMMENT;
         }
        /*
         * Check aborted
         */
         if ( *(cptr-1) == 'a' && *cptr == 'b' && *(cptr+1) == 'o') {
            if ( state == ON_COMMENT ) {
                state = ON_ABO;
            }
         }
         if ( *(cptr-1) == 'r' && *cptr == 't' && *(cptr+1) == 'e') {
            if ( state == ON_ABO) {
                aborted = 1;
            }
         }
        /*
         * Check resume line
         */
         if ( *(cptr-1) == 'r' && *cptr == 'e' && *(cptr+1) == 's') {
            if ( state == ON_COMMENT ) {
                state = ON_RES;
            }
         }
         if ( *(cptr-1) == 'u' && *cptr == 'm' && *(cptr+1) == 'e') {
            if ( state == ON_RES) {
                aborted = 0;
            }
         }

        /*
         * If data line... aborted is aborted
         */
         if ( *cptr == '\n' ) {
              next = *(cptr+1);
              if (isdigit(next) || next == '+' || next == '-' || next == '@') {
                  aborted = 0;
              }
         }
     }
     free(buffer);
     return(aborted);

/*
 * To be implemented
 *    - return 0  = not aborted
 *    - return 1  = aborted
 *    - return -1 = error
 *
 * implementation:  read whole scan
 *    - go to header after offset
 *    - read all till end of scan with size
 *    - search for a line with a) #C ( comment ) then "aborted"
 */
     return( 0 );
}
