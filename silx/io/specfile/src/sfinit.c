#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
/************************************************************************
 *
 *   File:          sfinit.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Initialization routines ( open/update/close )
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2005/05/25 13:01:32 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sfinit.c,v $
 *   Log: Revision 1.5  2005/05/25 13:01:32  sole
 *   Log: Back to revision 1.3
 *   Log:
 *   Log: Revision 1.3  2004/05/12 16:57:32  sole
 *   Log: windows support
 *   Log:
 *   Log: Revision 1.2  2002/11/12 13:23:43  sole
 *   Log: Version with added support for the new sf->updating flag
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 *   Log: Revision 2.2  2000/12/20 12:12:08  rey
 *   Log: bug corrected with SfAllMotors
 *   Log:
 * Revision 2.1  2000/07/31  19:04:42  19:04:42  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
/*
 *   File:         sfinit.c
 *
 *   Description:  This file implements basic routines on SPEC datafiles
 *                 SfOpen / SfClose / SfError
 *
 *                 SfUpdate is kept but it is obsolete
 *
 *   Version:      2.0
 *
 *   Date:         March 2000
 *
 *   Author:       Vicente REY
 *
 *   Copyright:    E.S.R.F. European Synchrotron Radiation Facility (c) 2000
 */
/*
 *   $Log: sfinit.c,v $
 *   Revision 1.5  2005/05/25 13:01:32  sole
 *   Back to revision 1.3
 *
 *   Revision 1.3  2004/05/12 16:57:32  sole
 *   windows support
 *
 *   Revision 1.2  2002/11/12 13:23:43  sole
 *   Version with added support for the new sf->updating flag
 *
 *   Revision 3.0  2000/12/20 14:17:19  rey
 *   Python version available
 *
 *   Revision 2.2  2000/12/20 12:12:08  rey
 *   bug corrected with SfAllMotors
 *
 * Revision 2.1  2000/07/31  19:04:42  19:04:42  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:27:19  13:27:19  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 *
 *
 *********************************************************************/
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <ctype.h>

#ifdef WIN32
#include <stdio.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

#include <SpecFile.h>
#include <SpecFileP.h>

/*
 * Defines
 */

#define ANY          0
#define NEWLINE      1
#define COMMENT      2

#define SF_ISFX      ".sfI"

#define SF_INIT      0
#define SF_READY     1
#define SF_MODIFIED  2

/*
 * Function declaration
 */

DllExport SpecFile * SfOpen   ( char *name,int *error);
DllExport SpecFile * SfOpen2  ( int fd, char *name,int *error);
DllExport int        SfClose  ( SpecFile *sf);
DllExport short      SfUpdate ( SpecFile *sf, int *error);
DllExport char     * SfError  ( int error);


#ifdef linux
char SF_SIGNATURE[] =  "Linux 2ruru Sf2.0";
#else
char SF_SIGNATURE[] =  "2ruru Sf2.0";
#endif

/*
 * Internal functions
 */
static short statusEnd     ( char c2, char c1);
static void  sfStartBuffer ( SpecFile *sf, SfCursor *cursor, short status,char c0, char c1,int *error);
static void  sfNewLine     ( SpecFile *sf, SfCursor *cursor, char c0,char c1,int *error);
static void  sfHeaderLine  ( SpecFile *sf, SfCursor *cursor, char c,int *error);
static void  sfNewBlock    ( SpecFile *sf, SfCursor *cursor, short how,int *error);
static void  sfSaveScan    ( SpecFile *sf, SfCursor *cursor, int *error);
static void  sfAssignScanNumbers (SpecFile *sf);
static void  sfReadFile    ( SpecFile *sf, SfCursor *cursor, int *error);
static void  sfResumeRead  ( SpecFile *sf, SfCursor *cursor, int *error);
#ifdef SPECFILE_USE_INDEX_FILE
static short sfOpenIndex   ( SpecFile *sf, SfCursor *cursor, int *error);
static short sfReadIndex   ( int sfi, SpecFile *sf, SfCursor *cursor, int *error);
static void  sfWriteIndex  ( SpecFile *sf, SfCursor *cursor, int *error);
#endif

/*
 * errors
 */
typedef struct _errors {
     int       code;
     char      *message;
} sf_errors ;

static
sf_errors errors[]={
{ SF_ERR_MEMORY_ALLOC     , "Memory allocation error ( SpecFile )"  },
{ SF_ERR_FILE_OPEN        , "File open error ( SpecFile )"      },
{ SF_ERR_FILE_CLOSE       , "File close error ( SpecFile )"         },
{ SF_ERR_FILE_READ        , "File read error ( SpecFile )"      },
{ SF_ERR_FILE_WRITE       , "File write error ( SpecFile )"      },
{ SF_ERR_LINE_NOT_FOUND   , "Line not found error ( SpecFile )"      },
{ SF_ERR_SCAN_NOT_FOUND   , "Scan not found error ( SpecFile )"      },
{ SF_ERR_HEADER_NOT_FOUND , "Header not found error ( SpecFile )"      },
{ SF_ERR_LABEL_NOT_FOUND  , "Label not found error ( SpecFile )"      },
{ SF_ERR_MOTOR_NOT_FOUND  , "Motor not found error ( SpecFile )"      },
{ SF_ERR_POSITION_NOT_FOUND    , "Position not found error ( SpecFile )" },
{ SF_ERR_LINE_EMPTY       , "Line empty or wrong data error ( SpecFile )"},
{ SF_ERR_USER_NOT_FOUND   , "User not found error ( SpecFile )"       },
{ SF_ERR_COL_NOT_FOUND    , "Column not found error ( SpecFile )"      },
{ SF_ERR_MCA_NOT_FOUND    , "Mca not found ( SpecFile )"      },
/* MUST be always the last one : */
{ SF_ERR_NO_ERRORS        , "OK ( SpecFile )"              },
};





/*********************************************************************
 *   Function:          SpecFile *SfOpen( name, error)
 *
 *   Description:       Opens connection to Spec data file.
 *                      Creates index list in memory.
 *
 *   Parameters:
 *              Input :
 *                      (1) Filename
 *              Output:
 *                      (2) error number
 *   Returns:
 *                      SpecFile pointer.
 *                      NULL if not successful.
 *
 *   Possible errors:
 *                      SF_ERR_FILE_OPEN
 *                      SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/

DllExport SpecFile *
SfOpen(char *name, int *error) {

   int         fd;
   fd   = open(name,SF_OPENFLAG);
   return (SfOpen2(fd, name, error));
}



/*********************************************************************
 *   Function:          SpecFile *SfOpen2( fd, name, error)
 *
 *   Description:       Opens connection to Spec data file.
 *                      Creates index list in memory.
 *
 *   Parameters:
 *              Input :
 *                      (1) Integer file handle
 *                      (2) Filename
 *              Output:
 *                      (3) error number
 *   Returns:
 *                      SpecFile pointer.
 *                      NULL if not successful.
 *
 *   Possible errors:
 *                      SF_ERR_FILE_OPEN
 *                      SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/

DllExport SpecFile *
SfOpen2(int fd, char *name,int *error) {
   SpecFile   *sf;
   short       idxret;
   SfCursor      cursor;
   struct stat mystat;

   if ( fd == -1 ) {
      *error = SF_ERR_FILE_OPEN;
      return ( (SpecFile *) NULL );
   }

  /*
   * Init specfile strucure
   */
#ifdef _WINDOWS
   static HANDLE hglb;
   hglb = GlobalAlloc(GPTR,sizeof(SpecFile));
   sf   = (SpecFile * ) GlobalLock(hglb);
#else
   sf = (SpecFile *) malloc ( sizeof(SpecFile ));
#endif
   stat(name,&mystat);

   sf->fd     = fd;
   sf->m_time = mystat.st_mtime;
   sf->sfname = (char *)strdup(name);

   sf->list.first      = (ObjectList *)NULL;
   sf->list.last       = (ObjectList *)NULL;
   sf->no_scans        = 0;
   sf->current         = (ObjectList *)NULL;
   sf->scanbuffer      = (char *)NULL;
   sf->scanheadersize  = 0;
   sf->filebuffer      = (char *)NULL;
   sf->filebuffersize  = 0;

   sf->no_labels       = -1;
   sf->labels          = (char **)NULL;
   sf->no_motor_names  = -1;
   sf->motor_names     = (char **)NULL;
   sf->no_motor_pos    = -1;
   sf->motor_pos       = (double *)NULL;
   sf->data            = (double **)NULL;
   sf->data_info       = (long *)NULL;
   sf->updating        = 0;

  /*
   * Init cursor
   */
   cursor.bytecnt      = 0;
   cursor.cursor       = 0;
   cursor.scanno       = 0;
   cursor.hdafoffset   = -1;
   cursor.dataoffset   = -1;
   cursor.mcaspectra   = 0;
   cursor.what         = 0;
   cursor.data         = 0;
   cursor.file_header  = 0;


#ifdef SPECFILE_USE_INDEX_FILE
  /*
   * Check if index file
   *   open it and continue from there
   */
   idxret = sfOpenIndex(sf,&cursor,error);
#else
   idxret = SF_INIT;
#endif

   switch(idxret) {
      case SF_MODIFIED:
          sfResumeRead(sf,&cursor,error);
          sfReadFile(sf,&cursor,error);
          break;

      case SF_INIT:
          sfReadFile(sf,&cursor,error);
          break;

      case SF_READY:
          break;

      default:
          break;
   }

   sf->cursor = cursor;

  /*
   * Once is all done assign scan numbers and orders
   */
   sfAssignScanNumbers(sf);

#ifdef SPECFILE_USE_INDEX_FILE
   if (idxret != SF_READY) sfWriteIndex(sf,&cursor,error);
#endif
   return(sf);
}




/*********************************************************************
 *
 *   Function:		int SfClose( sf )
 *
 *   Description:	Closes a file previously opened with SfOpen()
 *			and frees all memory .
 *   Parameters:
 *		Input:
 *			File pointer
 *   Returns:
 *			0 :  close successful
 *		       -1 :  errors occured
 *
 *********************************************************************/
DllExport int
SfClose( SpecFile *sf )
{
     register ObjectList  *ptr;
     register ObjectList  *prevptr;

     freeAllData(sf);

     for( ptr=sf->list.last ; ptr ; ptr=prevptr ) {
          free( (SpecScan *)ptr->contents );
          prevptr = ptr->prev;
	  free( (ObjectList *)ptr );
     }

     free ((char *)sf->sfname);
     if (sf->scanbuffer != NULL)
        free ((char *)sf->scanbuffer);

     if (sf->filebuffer != NULL)
        free ((char *)sf->filebuffer);

     if( close(sf->fd) ) {
	  return( -1 ) ;
     }

     free (  sf );
     sf    = (SpecFile *)NULL;

     return   (   0 );
}


/*********************************************************************
 *
 *   Function:          short SfUpdate( sf, error )
 *
 *   Description:       Updates connection to Spec data file .
 *                      Appends to index list in memory.
 *
 *   Parameters:
 *              Input :
 *                      (1) sf (pointer to the index list in memory)
 *              Output:
 *                      (2) error number
 *   Returns:
 *                      ( 0 ) => Nothing done.
 *                      ( 1 ) => File was updated
 *
 *   Possible errors:
 *                      SF_ERR_FILE_OPEN
 *                      SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/
DllExport short
SfUpdate ( SpecFile *sf, int *error )
{
    struct stat mystat;
    long   mtime;
   /*printf("In SfUpdate\n");
   __asm("int3");*/
    stat(sf->sfname,&mystat);

    mtime = mystat.st_mtime;

    if (sf->m_time != mtime)  {
       sfResumeRead (sf,&(sf->cursor),error);
       sfReadFile   (sf,&(sf->cursor),error);

       sf->m_time = mtime;
       sfAssignScanNumbers(sf);
#ifdef SPECFILE_USE_INDEX_FILE
       sfWriteIndex (sf,&(sf->cursor),error);
#endif
       return(1);
    }else{
       return(0);
    }
}


/*********************************************************************
 *
 *   Function:		char *SfError( code )
 *
 *   Description:	Returns the message associated with error 'code'.
 *
 *   Parameters:
 *		Input :	error code
 *
 *********************************************************************/
DllExport char *
SfError(int code ) {
     int	i;

     for ( i=0 ; errors[i].code!=0 ; i++ ) {
	     if ( errors[i].code == code ) break;
     }
     return( errors[i].message );
}


static void
sfReadFile(SpecFile *sf,SfCursor *cursor,int *error) {

   int         fd;

   char  *buffer,*ptr;

   long  size,bytesread;

   short  status;

   fd   = sf->fd;

   size = 1024*1024;


   if ( (buffer = (char *) malloc(size)) == NULL ) {
        /*
         * Try smaller buffer
         */
         size = 128 * 128;
         if ( (buffer = (char *) malloc(size)) == NULL ) {
              /*
               * Uhmmm
               */
              *error = SF_ERR_MEMORY_ALLOC;
              free(sf->sfname);
              free(sf);
              sf = (SpecFile *)NULL;
              return;
         }
   }

   status              = NEWLINE;
   while ((bytesread = read(fd,buffer,size)) > 0 ) {

      sfStartBuffer(sf,cursor,status,buffer[0],buffer[1],error);

      cursor->bytecnt++;
      for (ptr=buffer+1;ptr < buffer + bytesread -1; ptr++,cursor->bytecnt++) {
          if (*(ptr-1) == '\n' ) {
             sfNewLine(sf,cursor,*ptr,*(ptr+1),error);
          }
      }

      cursor->bytecnt++;
      status = statusEnd(buffer[bytesread-2],buffer[bytesread-1]);
  }

  free(buffer);

  sf->no_scans = cursor->scanno;
 /*
  * Save last
  */
  sfSaveScan(sf,cursor,error);

  return;

}


static void
sfResumeRead  ( SpecFile *sf, SfCursor *cursor, int *error) {
    cursor->bytecnt      = cursor->cursor;
    cursor->what         = 0;
    cursor->hdafoffset   = -1;
    cursor->dataoffset   = -1;
    cursor->mcaspectra   = 0;
    cursor->data         = 0;
    cursor->scanno--;
    sf->updating = 1;
    lseek(sf->fd,cursor->bytecnt,SEEK_SET);
    return;
}


#ifdef SPECFILE_USE_INDEX_FILE
static short
sfOpenIndex ( SpecFile *sf, SfCursor *cursor, int *error) {
    char *idxname;
    short namelength;
    int   sfi;

    namelength = strlen(sf->sfname) + strlen(SF_ISFX) + 1;

    idxname = (char *)malloc(sizeof(char) * namelength);

    sprintf(idxname,"%s%s",sf->sfname,SF_ISFX);

    if ((sfi = open(idxname,SF_OPENFLAG)) == -1) {
        free(idxname);
        return(SF_INIT);
    } else {
        free(idxname);
        return(sfReadIndex(sfi,sf,cursor,error));
    }
}


static short
sfReadIndex   ( int sfi, SpecFile *sf, SfCursor *cursor, int *error) {
    SfCursor   filecurs;
    char       buffer[200];
    long       bytesread,i=0;
    SpecScan   scan;
    short      modif = 0;
    long       mtime;

   /*
    * read signature
    */
    bytesread = read(sfi,buffer,sizeof(SF_SIGNATURE));
    if (strcmp(buffer,SF_SIGNATURE) || bytesread == 0 ) {
        return(SF_INIT);
    }

   /*
    * read cursor and specfile structure
    */
    if ( read(sfi,&mtime,   sizeof(long)) == 0)   return(SF_INIT);
    if ( read(sfi,&filecurs, sizeof(SfCursor)) == 0) return(SF_INIT);

    if (sf->m_time != mtime)  modif = 1;

    while(read(sfi,&scan, sizeof(SpecScan))) {
        addToList(&(sf->list), (void *)&scan, (long)sizeof(SpecScan));
        i++;
    }
    sf->no_scans = i;

    memcpy(cursor,&filecurs,sizeof(SfCursor));

    if (modif) return(SF_MODIFIED);

    return(SF_READY);
}


static void
sfWriteIndex  ( SpecFile *sf, SfCursor *cursor, int *error) {

    int         fdi;
    char       *idxname;
    short       namelength;
    ObjectList *obj;
    long        mtime;

    namelength = strlen(sf->sfname) + strlen(SF_ISFX) + 1;

    idxname = (char *)malloc(sizeof(char) * namelength);

    sprintf(idxname,"%s%s",sf->sfname,SF_ISFX);

    /* if ((fdi = open(idxname,SF_WRITEFLAG,SF_UMASK)) == -1) { */
    if ((fdi = open(idxname,O_CREAT | O_WRONLY,SF_UMASK)) == -1) {
        printf("    - cannot open. Error: (%d)\n",errno);
        free(idxname);
        return;
    } else {
        mtime = sf->m_time;
        write(fdi,SF_SIGNATURE,sizeof(SF_SIGNATURE));
 /*
  * Swap bytes for linux
  */
        write(fdi, (void *) &mtime, sizeof(long));
        write(fdi, (void *) cursor, sizeof(SfCursor));
        for( obj = sf->list.first; obj ; obj = obj->next)
           write(fdi,(void *) obj->contents, sizeof(SpecScan));
        close(fdi);
        free(idxname);
        return;
    }
}
#endif


/*****************************************************************************
 *
 *    Function:   static void sfStartBuffer()
 *
 *    Description:  start analyzing file buffer and takes into account the last
 *                  bytes of previous reading as defined in variable status
 *
 *****************************************************************************/
static void
sfStartBuffer(SpecFile *sf,SfCursor *cursor,short status,char c0,char c1,int *error) {

    if ( status == ANY ) {
        return;
    } else if ( status == NEWLINE ) {
        sfNewLine(sf,cursor,c0,c1,error);
    } else if ( status == COMMENT ) {
        cursor->bytecnt--;
        sfHeaderLine(sf,cursor,c0,error);
        cursor->bytecnt++;
    }

}


/*******************************************************************************
 *
 *    Function:   static void statusEnd()
 *
 *    Description:  ends analysis of file buffer and returns a variable
 *                  indicating staus ( last character is COMMENT,NEWLINE of ANY )
 *
 *******************************************************************************/
static short
statusEnd(char c2,char c1) {

      if (c2=='\n' && c1=='#') {
            return(COMMENT);
      } else if (c1=='\n') {
            return(NEWLINE);
      } else {
            return(ANY);
      }
}


static void
sfNewLine(SpecFile *sf,SfCursor *cursor,char c0,char c1,int *error) {
     if (c0 == '#') {
          sfHeaderLine(sf,cursor,c1,error);
     } else if (c0 == '@') {
          if ( cursor->data == 0 ) {
             cursor->dataoffset = cursor->bytecnt;
             cursor->data = 1;
          }
          cursor->mcaspectra++;
     } else if ( isdigit(c0) || c0 == '-' || c0 == '+' || c0 == ' ' || c0 == '\t') {
          if ( cursor->data == 0 ) {
              cursor->dataoffset = cursor->bytecnt;
              cursor->data = 1;
          }
     }
}


static void
sfHeaderLine(SpecFile *sf,SfCursor *cursor,char c,int *error) {
   if ( c == 'S') {
        sfNewBlock(sf,cursor,SCAN,error);
   } else if ( c == 'F') {
        sfNewBlock(sf,cursor,FILE_HEADER,error);
   } else {
        if (cursor->data && cursor->hdafoffset == -1 )
              cursor->hdafoffset = cursor->bytecnt;
   }
}


static void
sfNewBlock(SpecFile *sf,SfCursor *cursor,short newblock,int *error) {

  /*
   * Dispatch opened block
   */
   if (cursor->what == SCAN) {
        sfSaveScan(sf,cursor,error);
   } else if ( cursor->what == FILE_HEADER) {
        cursor->fileh_size = cursor->bytecnt - cursor->cursor + 1;
   }

  /*
   * Open new block
   */
   if (newblock == SCAN) {
        cursor->scanno++;
        cursor->what = SCAN;
   } else {
        cursor->file_header = cursor->bytecnt;
   }
   cursor->what         = newblock;
   cursor->hdafoffset   = -1;
   cursor->dataoffset   = -1;
   cursor->mcaspectra   = 0;
   cursor->data         = 0;
   cursor->cursor       = cursor->bytecnt;
}


static void
sfSaveScan(SpecFile *sf, SfCursor *cursor,int *error) {
    SpecScan  scan;
    SpecScan  *oldscan;
    register	ObjectList	*ptr;


    scan.index                 = cursor->scanno;
    scan.offset                = cursor->cursor;
    scan.size                  = cursor->bytecnt - cursor->cursor;
    scan.last                  = cursor->bytecnt - 1;
    scan.data_offset           = cursor->dataoffset;
    scan.hdafter_offset        = cursor->hdafoffset;
    scan.mcaspectra            = cursor->mcaspectra;
    scan.file_header           = cursor->file_header;

    if(sf->updating == 1){
        ptr = sf->list.last;
        oldscan=(SpecScan *)(ptr->contents);
        oldscan->index=scan.index;
        oldscan->offset=scan.offset;
        oldscan->size=scan.size;
        oldscan->last=scan.last;
        oldscan->data_offset=scan.data_offset;
        oldscan->hdafter_offset=scan.hdafter_offset;
        oldscan->mcaspectra=scan.mcaspectra;
        oldscan->file_header=scan.file_header;
        sf->updating=0;
    }else{
        addToList( &(sf->list), (void *)&scan, (long) sizeof(SpecScan));
    }
}


static void
sfAssignScanNumbers(SpecFile *sf) {

  int                    size,i;
  char                  *buffer,*ptr;

  char   buffer2[50];

  register   ObjectList *object,
                        *object2;
  SpecScan              *scan,
                        *scan2;

  size = 50;
  buffer = (char *) malloc(size);

  for ( object = (sf->list).first; object; object=object->next) {
        scan = (SpecScan *) object->contents;

        lseek(sf->fd,scan->offset,SEEK_SET);
        read(sf->fd,buffer,size);
        buffer[49] = '\0';

        for ( ptr = buffer+3,i=0; *ptr != ' ';ptr++,i++) buffer2[i] = *ptr;

        buffer2[i] = '\0';

        scan->scan_no = atol(buffer2);
        scan->order   = 1;
        for ( object2 = (sf->list).first; object2 != object; object2=object2->next) {
            scan2 = (SpecScan *) object2->contents;
            if (scan2->scan_no == scan->scan_no) scan->order++;
        }
  }
}

void
printCursor(SfCursor *cursor) {
   printf("<Cursor>\n");
   printf("   - Bytecnt:     %ld\n",cursor->bytecnt);
   printf("   - Cursor:      %ld\n",cursor->cursor);
   printf("   - Scanno:      %ld\n",cursor->scanno);
}
