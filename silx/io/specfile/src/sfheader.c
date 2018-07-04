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
 *   File:          sfheader.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Functions to access file and scan headers
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2002/11/20 09:01:29 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sfheader.c,v $
 *   Log: Revision 1.3  2002/11/20 09:01:29  sole
 *   Log: Added free(line); in SfTitle
 *   Log:
 *   Log: Revision 1.2  2002/11/14 16:18:48  sole
 *   Log: stupid bug removed
 *   Log:
 *   Log: Revision 1.1  2002/11/14 15:25:39  sole
 *   Log: Initial revision
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 * Revision 2.1  2000/07/31  19:05:09  19:05:09  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
#include <SpecFile.h>
#include <SpecFileP.h>

/*
 * Function Declaration
 */
DllExport char   * SfCommand        ( SpecFile *sf, long index, int *error );
DllExport long     SfNoColumns      ( SpecFile *sf, long index, int *error );
DllExport char   * SfDate           ( SpecFile *sf, long index, int *error );
DllExport double * SfHKL            ( SpecFile *sf, long index, int *error );

DllExport long     SfEpoch          ( SpecFile *sf, long index, int *error );
DllExport char   * SfUser           ( SpecFile *sf, long index, int *error );
DllExport char   * SfTitle          ( SpecFile *sf, long index, int *error );
DllExport char   * SfFileDate       ( SpecFile *sf, long index, int *error );
DllExport long     SfNoHeaderBefore ( SpecFile *sf, long index, int *error );
DllExport long     SfGeometry       ( SpecFile *sf, long index,
                                           char ***lines, int *error);
DllExport long     SfHeader         ( SpecFile *sf, long index, char *string,
                                           char ***lines, int *error);
DllExport long     SfFileHeader     ( SpecFile *sf, long index, char *string,
                                           char ***lines, int *error);

int  sfGetHeaderLine         ( SpecFile *sf, int from, char character,
                                             char **buf,int *error);
/*
 * Internal functions
 */
static char *sfFindWord      ( char *line, char *word, int *error );
static long  sfFindLines     ( char *from, char *to,char *string,
                                             char ***lines,int *error);
static char *sfOneLine       ( char *from, char *end, int *error);


/*********************************************************************
 *   Function:        char *SfCommand( sf, index, error )
 *
 *   Description:    Reads '#S' line ( without #S and scan number ).
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            String pointer,
 *            NULL => errors.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *            SF_ERR_FILE_READ
 *            SF_ERR_SCAN_NOT_FOUND
 *            SF_ERR_LINE_NOT_FOUND
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport char *
SfCommand( SpecFile *sf, long index, int *error )
{
     char    *ret_line=NULL;
     long     cnt,start,length;
     char    *ptr;

     /*
      * Choose scan
      */
     if (sfSetCurrent(sf,index,error) == -1)
          return(ret_line);

     cnt = 3;
     for ( ptr = sf->scanbuffer + cnt; *ptr != ' ' ; ptr++,cnt++);
     for ( ptr = sf->scanbuffer + cnt; *ptr == ' ' || *ptr == '\t'; ptr++,cnt++);

     start = cnt;
     for ( ptr = sf->scanbuffer + cnt; *ptr != '\n' ; ptr++,cnt++);

     length = cnt - start;

     /*
      * Return the rest .
      */
     ret_line = (char *) malloc ( sizeof(char) * ( length + 1) );
     if (ret_line == (char *)NULL) {
          *error = SF_ERR_MEMORY_ALLOC;
          return(ret_line);
     }

     ptr = sf->scanbuffer + start;
     memcpy(ret_line,ptr,sizeof(char) * length );
     ret_line[length] = '\0';

     return( ret_line );
}


/*********************************************************************
 *   Function:        long SfNoColumns( sf, index, error )
 *
 *   Description:    Gets number of columns in a scan
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Number of scan columns.(From #N line !)
 *            ( -1 ) if errors occured.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => readHeader()
 *            SF_ERR_LINE_NOT_FOUND
 *            SF_ERR_FILE_READ
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *********************************************************************/
DllExport long
SfNoColumns( SpecFile *sf, long index, int *error )
{
     long   col = -1;
     char  *buf=NULL;

     if ( sfSetCurrent(sf,index,error) == -1)
          return(-1);

     if ( sfGetHeaderLine( sf, FROM_SCAN, SF_COLUMNS, &buf, error) == -1)
          return(-1);
     col   = atol( buf );
     free(buf);
     return( col );
}


/*********************************************************************
 *   Function:        char *SfDate( sf, index, error )
 *
 *   Description:    Gets date from scan header
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Date.(From #D line !),
 *            NULL => errors.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => readHeader()
 *            SF_ERR_LINE_NOT_FOUND
 *            SF_ERR_FILE_READ
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport char *
SfDate(SpecFile *sf, long index, int *error )
{
     char        *line=NULL;

     if ( sfSetCurrent(sf,index,error) == -1 )
          return(line);

     if ( sfGetHeaderLine( sf, FROM_SCAN, SF_DATE, &line, error))
          return((char *)NULL);

     return( line );
}


/*********************************************************************
 *   Function:        double *SfHKL( sf, index, error )
 *
 *   Description:    Reads '#Q' line.
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Poiter to a 3x1 dbl. array( HKL[0]=HKL[H]=H_value,
 *                            HKL[1]=HKL[K]=K_value,
 *                            HKL[2]=HKL[L]=L_value.
 *            NULL => errors.
 *
 *   Possible errors:
 *            SF_ERR_LINE_EMPTY
 *            SF_ERR_FILE_READ
 *            SF_ERR_SCAN_NOT_FOUND
 *            SF_ERR_LINE_NOT_FOUND
 *            SF_ERR_MEMORY_ALLOC    | => mulstrtod()
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport double *
SfHKL( SpecFile *sf, long index, int *error )
{
     char        *line=NULL;
     double      *HKL = NULL;
     long         i;

     if ( sfSetCurrent(sf,index,error) == -1 )
          return((double *)NULL);

     if ( sfGetHeaderLine( sf, FROM_SCAN, SF_RECIP_SPACE, &line, error) == -1 )
          return((double *)NULL);

     /*
      * Convert into double .
      */
     i = mulstrtod( line, &HKL, error );
     free(line);

     if ( i < 0)
         return( (double *)NULL );

     if ( i != 3 ) {
        *error = SF_ERR_LINE_EMPTY;
         free( HKL );
         return( (double *)NULL );
     }

     return( HKL );
}


/*********************************************************************
 *   Function:        long SfEpoch( sf, index, error )
 *
 *   Description:     Gets epoch from the last file header.
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *                   (2) Index
 *        Output:
 *                   (3) error number
 *   Returns:
 *            Epoch.(From #E line !)
 *            ( -1 ) if errors occured.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    | => readHeader()
 *            SF_ERR_LINE_NOT_FOUND
 *            SF_ERR_FILE_READ
 *            SF_ERR_HEADER_NOT_FOUND
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *********************************************************************/
DllExport long
SfEpoch( SpecFile *sf, long index, int *error )
{
     char   *buf=NULL;
     long   epoch = -1;

     if ( sfSetCurrent(sf,index,error) == -1 )
          return(-1);

     if ( sfGetHeaderLine(sf,FROM_FILE,SF_EPOCH,&buf,error) == -1 )
          return(-1);

     epoch  = atol( buf );
     free(buf);

     return( epoch );
}


/*********************************************************************
 *   Function:        char SfFileDate( sf, index, error )
 *
 *   Description:     Gets date from the last file header
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *                   (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Date.(From #D line !)
 *            NULL => errors.
 *
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC   | => readHeader()
 *            SF_ERR_LINE_NOT_FOUND
 *            SF_ERR_LINE_EMPTY
 *            SF_ERR_FILE_READ
 *            SF_ERR_HEADER_NOT_FOUND
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *********************************************************************/
DllExport char *
SfFileDate( SpecFile *sf, long index, int *error )
{
     char   *date = NULL;

     if ( sfSetCurrent(sf,index,error) == -1 )
          return((char *)NULL);

     if ( sfGetHeaderLine(sf,FROM_FILE,SF_DATE,&date,error) == -1 )
          return((char *)NULL);

     return( date );
}


/*********************************************************************
 *   Function:        long SfNoHeaderBefore( sf, index, error )
 *
 *   Description:    Gets number of scan header lines before data.
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Scan index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Number of scan header lines before data ,
 *            ( -1 ) => errors.
 *   Possible errors:
 *            SF_ERR_SCAN_NOT_FOUND
 *
 *********************************************************************/
DllExport long
SfNoHeaderBefore( SpecFile *sf, long index, int *error )
{
     if ( sfSetCurrent(sf,index,error) == -1 )
          return(-1);

    /*
     * Obsolete... give some reasonable!
     */
     return(-1);
}


/*********************************************************************
 *   Function:        char *SfUser( sf, index, error )
 *
 *   Description:    Gets spec user information from the last file header
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            User.(From 1st #C line !)
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC    ||=>  findWordInLine()
 *            SF_ERR_LINE_NOT_FOUND    |
 *            SF_ERR_FILE_READ    |
 *            SF_ERR_SCAN_NOT_FOUND    | =>  getFirstFileC()
 *            SF_ERR_HEADER_NOT_FOUND    |
 *            SF_ERR_USER_NOT_FOUND
 *
 *********************************************************************/
DllExport char *
SfUser( SpecFile *sf, long index, int *error )
{

     char        *line=NULL;
     char        *user;
     char         word[] = "User =";

     if (sfSetCurrent(sf,index,error) == -1)
          return((char *)NULL);

     if (sfGetHeaderLine( sf, FROM_FILE, SF_COMMENT, &line, error) == -1)
          return((char *)NULL);

     /*
      * Find user.
      */
     user = sfFindWord( line, word, error );

     if ( user == (char *) NULL) {
        *error = SF_ERR_USER_NOT_FOUND;
         return((char *)NULL);
     }

     free(line);
     return( user );
}


/*********************************************************************
 *   Function:        long SfTitle( sf, index, error )
 *
 *   Description:    Gets spec title information from the last file header
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) Index
 *        Output:
 *            (3) error number
 *   Returns:
 *            Title.(From 1st #C line !)
 *            NULL => errors.
 *   Possible errors:
 *            SF_ERR_LINE_EMPTY
 *            SF_ERR_MEMORY_ALLOC
 *            SF_ERR_LINE_NOT_FOUND    |
 *            SF_ERR_FILE_READ    |
 *            SF_ERR_SCAN_NOT_FOUND    | =>  getFirstFileC()
 *            SF_ERR_HEADER_NOT_FOUND    |
 *
 *********************************************************************/
DllExport char *
SfTitle( SpecFile *sf, long index, int *error )
{
     char        *line=NULL;
     char        *title;
     char        *ptr;
     long         i;

     if (sfSetCurrent(sf,index,error) == -1)
          return((char *)NULL);

     if (sfGetHeaderLine( sf, FROM_FILE, SF_COMMENT, &line, error) == -1)
          return((char *)NULL);

     /*
      * Get title.( first word )
      */
     ptr = line;

     for ( i=0,ptr=line ; *ptr!='\t' && *ptr!='\n' && *ptr!='\0' ; i++ ) {
        if ( *ptr==' ' ) {
              if ( *(++ptr)==' ' ) {
                 break;
              } else ptr--;
        }
        ptr++;
     }

     if ( i==0 ) {
      *error = SF_ERR_LINE_EMPTY;
       return( (char *)NULL );
     }

     title = (char *)malloc( sizeof(char) * ( i+1 ) );

     if ( title == (char *)NULL ) {
       *error = SF_ERR_MEMORY_ALLOC;
        return( title );
     }

     memcpy( title, line, sizeof(char) * i  );
     /* Next line added by Armando, it may be wrong */
     free(line);
     title[i] = '\0';

     return( title );
}


DllExport long
SfGeometry ( SpecFile *sf, long index, char ***lines, int *error)
{
    char string[] = " \0";

    string[0] = SF_GEOMETRY;

    return(SfHeader(sf,index,string,lines,error));
}


DllExport long
SfHeader ( SpecFile *sf, long index, char *string, char ***lines, int *error)
{
     char   *headbuf,
            *endheader;

     long nb_found;

     if (sfSetCurrent(sf,index,error) == -1)
          return(-1);

     headbuf   = sf->scanbuffer;
     endheader = sf->scanbuffer + sf->scansize;

     nb_found = sfFindLines(headbuf, endheader,string, lines,error);

     if (nb_found == 0) {
          return SfFileHeader(sf,index,string,lines,error);
     } else {
          return nb_found;
     }
}



DllExport long
SfFileHeader ( SpecFile *sf, long index, char *string, char ***lines, int *error)
{
     char   *headbuf,
            *endheader;

     if (sfSetCurrent(sf,index,error) == -1)
          return(-1);
     if (sf->filebuffersize > 0)
     {
        headbuf   = sf->filebuffer;
        endheader = sf->filebuffer + sf->filebuffersize;

        return(sfFindLines(headbuf,endheader,string,lines,error));
     }
     else
     {
         return 0;
     }
}


static long
sfFindLines(char *from,char *to,char *string,char ***ret,int *error)
{
     char  **lines;
     long    found;
	 unsigned long j;
     char   *ptr;
     short   all=0;

     found = 0;
     ptr   = from;

     if ( string == (char *) NULL || strlen(string) == 0)
           all = 1;

     /*
      * Allocate memory for an array of strings
      */
     if ( (lines = (char **)malloc( sizeof(char *) )) == (char **)NULL ) {
        *error = SF_ERR_MEMORY_ALLOC;
         return ( -1 );
     }

     /*
      * First line
      */
     if ( ptr[0] == '#' ) {
        if ( all ) {
           lines = (char **) realloc ( lines, (found+1) * sizeof(char *) );
           lines[found] = sfOneLine(ptr,to,error);
           found++;
        } else if ( ptr[1] == string[0]) {
           for ( j=0; j < strlen(string) && ptr+j< to;j++)
               if ( ptr[j+1] != string[j]) break;
           if ( j == strlen(string)) {
                lines = (char **) realloc ( lines, (found+1) * sizeof(char *) );
                lines[found] = sfOneLine(ptr,to,error);
                found++;
           }
        }
     }

    /*
     * The rest
     */
     for ( ptr = from + 1;ptr < to - 1;ptr++) {
         if ( *(ptr - 1) == '\n' && *ptr == '#' ) {
              if ( all ) {
                 lines = (char **) realloc ( lines, (found+1) * sizeof(char *) );
                 lines[found] = sfOneLine(ptr,to,error);
                 found++;
              } else if ( *(ptr+1) == string[0]) {
                 for ( j=0; j < strlen(string) && (ptr + j) < to;j++)
                        if ( ptr[j+1] != string[j]) break;
                 if ( j == strlen(string)) {
                    lines = (char **) realloc ( lines, (found+1) * sizeof(char *) );
                    lines[found] = sfOneLine(ptr,to,error);
                    found++;
                 }
              }
         }
     }

     if (found) *ret = lines;
     else free(lines);

     return(found);
}


/*********************************************************************
 *   Function:       char *sfGetHeaderLine( SpecFile *sf, sf_char, end, error )
 *
 *   Description:    Gets one '#sf_char' line.
 *
 *   Parameters:
 *        Input :    (1) File pointer
 *            (2) sf_character
 *            (3) end ( where to stop the search )
 *        Output:
 *            (4) error number
 *   Returns:
 *            Pointer to the line ,
 *            NULL in case of errors.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *            SF_ERR_FILE_READ    | => findLine()
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
int
sfGetHeaderLine( SpecFile *sf, int from , char sf_char, char **buf, int *error)
{

     char   *ptr,*headbuf;
     char   *endheader;
     int     found;

     found = 0;

     if ( from == FROM_SCAN ) {
         headbuf   = sf->scanbuffer;
         endheader = sf->scanbuffer + sf->scanheadersize;
     } else if ( from == FROM_FILE ) {
         if ( sf->filebuffersize == 0 ) {
             *error = SF_ERR_LINE_NOT_FOUND;
              return(-1);
         }
         headbuf   = sf->filebuffer;
         endheader = sf->filebuffer + sf->filebuffersize;
     } else {
         *error = SF_ERR_LINE_NOT_FOUND;
         return(-1);
     }

     if ( headbuf[0] == '#' && headbuf[1] == sf_char) {
        found = 1;
        ptr   = headbuf;
     } else {
        for ( ptr = headbuf + 1;ptr < endheader - 1;ptr++) {
           if ( *(ptr - 1) == '\n' && *ptr == '#' && *(ptr+1) == sf_char) {
                found = 1;
                break;
           }
        }
     }

     if (!found) {
         *error = SF_ERR_LINE_NOT_FOUND;
         return(-1);
     }

    /*
     * Beginning of the thing after '#X '
     */
     ptr = ptr + 3;

     *buf = sfOneLine(ptr,endheader,error);

     return( 0 );
}

static char *
sfOneLine(char *from,char *end,int *error)
{
     static char linebuf[5000];

     char *ptr,*buf;
     long  i;

     ptr = from;

     for(i=0;*ptr != '\n' && ptr < end;ptr++,i++) {
         linebuf[i] = *ptr;
     }

     linebuf[i]='\0';

     buf = (char *) malloc ( i+1 );

     if (buf == ( char * ) NULL ) {
        *error = SF_ERR_MEMORY_ALLOC;
         return((char *)NULL);
     }
     strcpy(buf,(char *)linebuf);

     return(buf);
}


/*********************************************************************
 *   Function:        char *sfFindWord( line, word, error )
 *
 *   Description:    Looks for 'word' in given line and returns a
 *                   copy of the rest of the line after the found word .
 *
 *   Parameters:
 *        Input :    (1) Line pointer
 *            (2) Word pointer
 *        Output:
 *            (3) error number
 *   Returns:
 *            Rest of the line after word.
 *            NULL => not found.
 *   Possible errors:
 *            SF_ERR_MEMORY_ALLOC
 *
 *********************************************************************/
static char *
sfFindWord( char *line, char *word, int *error )
{
     char    *ret;

     line = strstr( line, word );

     if ( line == (char *)NULL ) {
         return( line );
     }

     line += strlen( word );

     /*
      * Delete blanks.
      */
     while ( *line == ' ' || *line == '\t' ) line++;

     /*
      * Copy the rest.
      */
     ret = (char *)malloc( sizeof(char) * ( 1 + strlen( line )) );

     if ( ret == (char *)NULL ) {
          *error = SF_ERR_MEMORY_ALLOC;
          return(ret);
     }

     memcpy( ret, line, sizeof(char) * ( 1 + strlen( line )) );

     return( ret );
}

