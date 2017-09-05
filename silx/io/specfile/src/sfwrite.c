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
 *   File:          sfwrite.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Functions for scan output
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2003/09/12 13:20:35 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sfwrite.c,v $
 *   Log: Revision 1.1  2003/09/12 13:20:35  rey
 *   Log: Initial revision
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 * Revision 2.1  2000/07/31  19:05:14  19:05:14  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
#include <SpecFile.h>
#include <SpecFileP.h>
#ifndef WIN32
#include <unistd.h>
#endif
/*
 * Declarations
 */
DllExport SpecFileOut  *SfoInit  ( SpecFile *sf, int *error );
DllExport void      SfoClose     ( SpecFileOut *sfo );
DllExport long      SfoSelectAll ( SpecFileOut *sfo, int *error );
DllExport long      SfoSelectOne ( SpecFileOut *sfo, long index,
                                                int *error );
DllExport long      SfoSelect    ( SpecFileOut *sfo, long *list,
                                                int *error );
DllExport long      SfoSelectRange ( SpecFileOut *sfo, long begin,
                                                long end, int *error );
DllExport long      SfoRemoveOne ( SpecFileOut *sfo, long index,
                                                int *error );
DllExport long      SfoRemove    ( SpecFileOut *sfo, long *list,
                                                int *error );
DllExport long      SfoRemoveRange ( SpecFileOut *sfo, long begin,
                                                long end, int *error );
DllExport long      SfoRemoveAll ( SpecFileOut *sfo, int *error );
DllExport long      SfoWrite     ( SpecFileOut *sfo, char *name,
                                                int *error );
DllExport long      SfoGetList   ( SpecFileOut *sfo, long **list,
                                                int *error );

/*
 * Internal functions
 */
static int sfoWriteOne(SpecFileOut *sfo,int output, long index,int *error);


/*********************************************************************
 *   Function:        SpecFileOut *SfoInit( sf, error )
 *
 *   Description:    Initializes a SpecFileOut structure:
 *                - pointer to SpecFile
 *                - list of scans to be copied
 *                - size of this list
 *                - last written file header
 *   Parameters:
 *        Input :    (1) SpecFile pointer
 *
 *        Output:
 *            (2) error number
 *   Returns:
 *            Pointer to the initialized SpecFileOut structure.
 *            NULL in case of an error.
 *
 *   Possible errors:
 *            SF_ERR_MEMOREY_ALLOC
 *
 *   Remark:    This function MUST be the FIRST called before
 *        any other WRITE function is called !
 *
 *********************************************************************/
DllExport SpecFileOut *
SfoInit( SpecFile *sf, int *error )
{
     SpecFileOut    *sfo;

     /*
      * Alloc memory
      */
     sfo = (SpecFileOut *) malloc ( sizeof(SpecFileOut) );

     if ( sfo == (SpecFileOut *)NULL ) {
         *error = SF_ERR_MEMORY_ALLOC;
         return( (SpecFileOut *)NULL );
     }

     /*
      * Initialize
      */
     sfo->sf          = sf;
     sfo->list        = (long *)NULL;
     sfo->list_size   =  0;
     sfo->file_header = -1;

     return( sfo );
}


/*********************************************************************
 *   Function:        long  SfoGetList( sfo, list, error )
 *
 *   Description:     Makes a copy of the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *
 *        Output: (2) Copy of the output list of spec scan indices.
 *            (3) error code
 *   Returns:
 *            Number of scan indices in the output list ,
 *            (  0 ) => list empty( (long *)NULL ) ), no errors
 *            ( -1 ) in case of an error.
 *
 *   Possible errors:
 *            SF_ERR_MEMOREY_ALLOC
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long
SfoGetList( SpecFileOut *sfo, long **list, int *error )
{
     long    i;

     *list = (long *)NULL;

     if ( sfo->list_size > 0 ) {
      *list = (long *)malloc( sfo->list_size * sizeof(long) );
      if ( *list == (long *)NULL ) {
           *error = SF_ERR_MEMORY_ALLOC;
           return( -1 );
      }
      for ( i=0 ; i < sfo->list_size ; i++ ) {
           (*list)[i] = sfo->list[i];
      }
     } else *list = (long *)NULL;

     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoSelectOne( sfo, index, error )
 *
 *   Description:    Adds one scan index to the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) Scan index
 *        Output:
 *            (3) error code
 *   Returns:
 *            ( -1 ) => error
 *            Number of scan indices in the SpecFileOut list.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/
DllExport long
SfoSelectOne( SpecFileOut *sfo, long index, int *error )
{
     long    i;

     /*
      * Check if index exists or if it's out of range.
      */
     if ( index > sfo->sf->no_scans || index < 1 ) {
        return( sfo->list_size );
     }

     /*
      * Alloc memory for the new index and add it to the list.
      */
     if ( sfo->list == (long *)NULL ) {
      sfo->list = (long *)malloc( sizeof(long) );
      if ( sfo->list == (long *)NULL ) {
           *error = SF_ERR_MEMORY_ALLOC;
           return( -1 );
      }
      sfo->list_size = 1;
     } else {
      /*
       * Is the new index already in list ?
       */
      for ( i=0 ; i<sfo->list_size ; i++ )
        if ( index == sfo->list[i] ) return( sfo->list_size );
      sfo->list = realloc( sfo->list, ++(sfo->list_size) * sizeof(long) );
      if ( sfo->list == (long *)NULL ) {
           *error = SF_ERR_MEMORY_ALLOC;
           sfo->list_size = 0;
           return( -1 );
      }
     }
     sfo->list[sfo->list_size-1] = index;
     printf("Adding scan %ld\n",index);

     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoSelect( sfo, list, error )
 *
 *   Description:    Adds several scan indices to the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) List scan indices (!The last element
 *                        MUST be a '0' !)
 *        Output:
 *            (3) error code
 *   Returns:
 *            ( -1 ) => error
 *            Number of scan indices in the SpecFileOut list.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => SfoSelectOne()
 *
 *********************************************************************/
DllExport long
SfoSelect( SpecFileOut *sfo, long *list, int *error )
{
     for ( ; *list != 0 ; list++ ) {
      if ( SfoSelectOne( sfo, *list , error ) < 0 ) return( -1 );
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoSelectRange( sfo, begin, end, error )
 *
 *   Description:    Adds scan indices between 'begin' and 'end'
 *            to the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) First ...
 *            (3) Last index to be added
 *        Output:
 *            (4) error code
 *   Returns:
 *            ( -1 ) => error
 *            Number of scan indices in the SpecFileOut list.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => SfoSelectOne()
 *
 *********************************************************************/
DllExport long
SfoSelectRange( SpecFileOut *sfo, long begin, long end, int *error )
{
     long    i;

     if ( begin > end ) {
      i=begin;
      begin = end;
      end = i;
     }
     if ( begin < 1 || end > sfo->sf->no_scans ) {
      return( sfo->list_size );
     }
     for ( i=begin ; i<=end ; i++ ) {
      if ( SfoSelectOne( sfo, i , error ) < 0 ) return( -1 );
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoSelectAll( sfo, error )
 *
 *   Description:    Writes all scan indices in the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOutput pointer
 *        Output:    (2) error number
 *   Returns:
 *            ( -1 ) => error
 *            Number of scan indices in the SpecFileOut list.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/
DllExport long
SfoSelectAll( SpecFileOut *sfo, int *error )
{
     long    i;

     if ( sfo->sf->no_scans > 0 ) {
      for ( i=1 ; i<=sfo->sf->no_scans ; i++ ) {
           if ( SfoSelectOne( sfo, i , error ) < 0 ) return( -1 );
      }
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoRemoveOne( sfo, index, error )
 *
 *   Description:    Removes one scan index from the SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) Scan index to be removed
 *        Output:
 *            (3) error code
 *   Returns:
 *            Number of scans left ,
 *            (  0 ) => list empty( (long *)NULL ) ), no errors
 *            ( -1 ) => error.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/
DllExport long
SfoRemoveOne( SpecFileOut *sfo, long index, int *error )
{
     long    i;
     int    found = 0;

     /*
      * Look for scan index and delete.
      */
     for ( i=0 ; i < (sfo->list_size - found) ; i++ ) {
      if ( sfo->list[i] == index ) found = 1;
      if ( found ) sfo->list[i]=sfo->list[i+1];
     }

     /*
      * Free unused memory
      */
     if ( found ) {
      (sfo->list_size)--;
      sfo->list = realloc( sfo->list, sfo->list_size * sizeof(long) );
      if ( sfo->list == (long *)NULL && sfo->list_size != 0 ) {
           *error = SF_ERR_MEMORY_ALLOC;
           return( -1 );
      }
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoRemove( sfo, list, error )
 *
 *   Description:    Removes several scans indices from the
 *                        SpecFileOut list.
 *
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) List of scan indices to be removed
 *                ( !!! The last element MUST be a '0' !!! )
 *        Output:
 *            (3) error code
 *   Returns:
 *            Number of scan indices left ,
 *            (  0 ) => list empty( (long *)NULL ) ), no errors
 *            ( -1 ) => error.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => SfoRemoveOne()
 *
 *********************************************************************/
DllExport long
SfoRemove( SpecFileOut *sfo, long *list, int *error )
{
     for ( ; *list != 0 ; list++ ) {
      if ( SfoRemoveOne( sfo, *list , error ) < 0 ) return( -1 );
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoRemoveRange( sfo, begin, end, error )
 *
 *   Description:    Removes scans indices from 'begin' to 'end'
 *                    from the SpecFileOut list.
 *
 *   Parameters:
 *        Input :
 *            (1) SpecFileOut pointer
 *            (2) First ...
 *            (3) Last index to be removed
 *        Output:
 *            (4) error code
 *   Returns:
 *            Number of scan indices left ,
 *            (  0 ) => list empty( (long *)NULL ) ), no errors
 *            ( -1 ) => error.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => SfoRemoveOne()
 *
 *********************************************************************/
DllExport long
SfoRemoveRange( SpecFileOut *sfo, long begin, long end, int *error )
{
     long    i;

     if ( begin > end ) {
      i=begin;
      begin = end;
      end = i;
     }
     if ( begin < 1 || end > sfo->sf->no_scans ) {
      return( sfo->list_size );
     }
     for ( i=begin ; i <= end ; i++ ) {
      if ( SfoRemoveOne( sfo, i, error ) < 0 ) return( -1 );
     }
     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        long SfoRemoveAll( sfo, error )
 *
 *   Description:    Removes all scans indices
 *                    from the SpecFileOut list.
 *
 *   Parameters:
 *        Input :
 *            (1) SpecFileOut pointer
 *        Output:
 *            (2) error code
 *   Returns:
 *            ( 0 ) => OK
 *
 *********************************************************************/
DllExport long
SfoRemoveAll( SpecFileOut *sfo, int *error )
{
     free( sfo->list );
     sfo->list        = (long *)NULL;
     sfo->list_size        =  0;
     sfo->file_header      = -1;
     return( 0 );
}


/*********************************************************************
 *   Function:        int SfoWrite( sfo, name, error )
 *
 *   Description:    Writes (appends) SpecScans specified in the sfo->list
 *            in the file 'name'. Related file headers are copied
 *            too.
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *            (2) Output file name
 *        Output:
 *            (3) error number
 *   Returns:
 *            Number of written scans,
 *            (-1 ) => Errors occured
 *   Possible errors:
 *            SF_ERR_FILE_WRITE    | => cpyBlock()
 *            SF_ERR_FILE_READ
 *            SF_ERR_FILE_OPEN
 *            SF_ERR_FILE_CLOSE
 *
 *********************************************************************/
DllExport long
SfoWrite( SpecFileOut *sfo, char *name, int *error )
{
     int     output;
     long     i;

     if ( sfo == (SpecFileOut *)NULL || sfo->list_size<1 ) return( 0 );

     /*
      * Open file
      */
     if ( (output = open(name, O_CREAT | O_RDWR | O_APPEND, SF_UMASK )) == (int)NULL ) {
        *error = SF_ERR_FILE_OPEN;
         return( -1 );
     }

     for ( i=0 ; i < sfo->list_size ; i++ )
          sfoWriteOne(sfo,output,sfo->list[i],error);

     if ( close( output ) ) {
          *error = SF_ERR_FILE_CLOSE;
           return( -1 );
     }

     return( sfo->list_size );
}


/*********************************************************************
 *   Function:        int SfoClose( sfo )
 *
 *   Description:    Frees all memory used by
 *            SpecFileOut structure.
 *   Parameters:
 *        Input :    (1) SpecFileOut pointer
 *
 *   Remark:    This function should be called after all
 *                     writing operations.
 *
 *********************************************************************/
DllExport void
SfoClose( SpecFileOut *sfo )
{
     /*
      * Free memory.
      */
     free( sfo->list );
     free( sfo );
}


static int
sfoWriteOne(SpecFileOut *sfo,int output,long index,int *error)
{
   long file_header,size;
   SpecFile *sf;

   if ( sfSetCurrent(sfo->sf,index,error) == -1 ) {
       *error = SF_ERR_SCAN_NOT_FOUND;
        return(-1);
   }

  /*
   * File header
   */
   sf = sfo->sf;

   file_header = ((SpecScan *)sf->current->contents)->size;

   if (file_header != -1  && file_header != sfo->file_header ) {
        printf("Writing %ld bytes\n",sf->filebuffersize);
        write(output, (void *) sf->filebuffer, sf->filebuffersize);
        sfo->file_header = file_header;
   }

  /*
   * write scan
   */
   size = ((SpecScan *)sf->current->contents)->size;

   if ( write(output,(void *) sf->scanbuffer,size) == -1 ) {
       *error = SF_ERR_FILE_WRITE;
        return(-1);
   }
   return(0);
}
