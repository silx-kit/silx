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
/***************************************************************************
 *
 *  File:            SpecFile.h
 *
 *  Description:     Include file for treating spec data files.
 *
 *  Author:          Vicente Rey
 *
 *  Created:         2 March 1995
 *
 *    (copyright by E.S.R.F.  March 1995)
 *
 ***************************************************************************/
#ifndef SPECFILE_H
#define SPECFILE_H

#include <math.h>
#include <stdio.h>
#include <fcntl.h>

#include <time.h>
#include <stdlib.h>
/* #include <malloc.h> */
#include <string.h>
#include <Lists.h>

#ifdef _WINDOWS   /* compiling on windows */
#include <windows.h>
#include <io.h>
#define SF_OPENFLAG   O_RDONLY | O_BINARY
#define SF_WRITEFLAG  O_CREAT | O_WRONLY
#define SF_UMASK      0666
#else   /* if not windows */
#define SF_OPENFLAG   O_RDONLY
#define SF_WRITEFLAG  O_CREAT | O_WRONLY
#define SF_UMASK      0666
#endif

#ifdef _GENLIB    /* for windows dll generation */
#define DllExport __declspec (dllexport)
#else
#define DllExport
#endif


#ifdef SUN4
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

/*
 * Defines.
 */
#define  ROW            0  /* data_info index for no. of data rows  */
#define  COL            1  /* data_info index for no. of data columns*/
#define  REG            2  /* data_info index for regular            */

#define  H              0
#define  K              1
#define  L              2
#define  ABORTED       -1
#define  NOT_ABORTED    0

#define  SF_ERR_NO_ERRORS           0
#define  SF_ERR_MEMORY_ALLOC        1
#define  SF_ERR_FILE_OPEN           2
#define  SF_ERR_FILE_CLOSE          3
#define  SF_ERR_FILE_READ           4
#define  SF_ERR_FILE_WRITE          5
#define  SF_ERR_LINE_NOT_FOUND      6
#define  SF_ERR_SCAN_NOT_FOUND      7
#define  SF_ERR_HEADER_NOT_FOUND    8
#define  SF_ERR_LABEL_NOT_FOUND     9
#define  SF_ERR_MOTOR_NOT_FOUND     10
#define  SF_ERR_POSITION_NOT_FOUND  11
#define  SF_ERR_LINE_EMPTY          12
#define  SF_ERR_USER_NOT_FOUND      13
#define  SF_ERR_COL_NOT_FOUND       14
#define  SF_ERR_MCA_NOT_FOUND       15

typedef struct _SfCursor {
    long  int scanno;      /* nb of scans */
    long  int cursor;      /* beginning of current scan */
    long  int hdafoffset;  /* global offset of header after beginning of data */
    long  int datalines;   /* contains nb of data lines */
    long  int dataoffset;  /* contains data offset from begin of scan */
    long  int mcaspectra;  /* contains nb of mca spectra in scan */
    long  int bytecnt;     /* total file byte count */
    long  int what;        /* scan of file block */
    long  int data;        /* data flag */
    long  int file_header; /* address of file header for this scan */
    long  int fileh_size;  /* size of it */
} SfCursor;


typedef struct _SpecFile{
  int             fd;
  long            m_time;
  char           *sfname;
  struct _ListHeader    list;
  long int        no_scans;
  ObjectList     *current;
  char           *scanbuffer;
  long            scanheadersize;
  char           *filebuffer;
  long            filebuffersize;
  long            scansize;
  char          **labels;
  long int        no_labels;
  char          **motor_names;
  long int        no_motor_names;
  double         *motor_pos;
  long int        no_motor_pos;
  double        **data;
  long           *data_info;
  SfCursor        cursor;
  short           updating;
} SpecFile;

typedef struct _SpecFileOut{
  SpecFile        *sf;
  long            *list;
  long             list_size;
  long             file_header;
} SpecFileOut;

typedef struct _SpecScan {
  long int        index;
  long int        scan_no;
  long int        order;
  long int        offset;
  long int        size;
  long int        last;
  long int        file_header;
  long int        data_offset;
  long int        hdafter_offset;
  long int        mcaspectra;
} SpecScan;

/*
 * Function declarations.
 */

   /*
    * Init
    */
/*
 * init
 */
DllExport extern    SpecFile  *SfOpen        ( char *name, int *error );
DllExport extern    short      SfUpdate      ( SpecFile *sf,int *error );
DllExport extern    int        SfClose       ( SpecFile *sf );

/*
 * indexes
 */
DllExport extern    long    SfScanNo      ( SpecFile *sf );
DllExport extern    long   *SfList        ( SpecFile *sf, int *error );
DllExport extern    long    SfCondList    ( SpecFile *sf, long cond,
                                                long **scan_list, int *error );
DllExport extern    long    SfIndex       ( SpecFile *sf, long number,
                                                long order );
DllExport extern    long    SfIndexes     ( SpecFile *sf, long number,
                                                long **indexlist );
DllExport extern    long    SfNumber      ( SpecFile *sf, long index );
DllExport extern    long    SfOrder       ( SpecFile *sf, long index );
DllExport extern    int     SfNumberOrder ( SpecFile *sf, long index,
                                                long *number, long *order );

   /*
    * Header
    */
DllExport extern    char   *SfCommand        ( SpecFile *sf, long index, int *error );
DllExport extern    long    SfNoColumns      ( SpecFile *sf, long index, int *error );
DllExport extern    char   *SfDate           ( SpecFile *sf, long index, int *error );
DllExport extern    long    SfEpoch          ( SpecFile *sf, long index, int *error );
DllExport extern    long    SfNoHeaderBefore ( SpecFile *sf, long index, int *error );
DllExport extern    double *SfHKL            ( SpecFile *sf, long index, int *error );
DllExport extern    long    SfHeader         ( SpecFile *sf, long index, char    *string,
                                          char ***lines, int *error );
DllExport extern    long    SfGeometry       ( SpecFile *sf, long index,
                                          char ***lines, int *error );
DllExport extern    long    SfFileHeader     ( SpecFile *sf, long index, char *string,
                                          char ***lines, int *error );
DllExport extern    char   *SfFileDate       ( SpecFile *sf, long index, int *error );
DllExport extern    char   *SfUser           ( SpecFile *sf, long index, int *error );
DllExport extern    char   *SfTitle          ( SpecFile *sf, long index, int *error );

   /*
    * Labels
    */
DllExport extern    long    SfAllLabels      ( SpecFile *sf, long index,
                                              char ***labels, int *error );
DllExport extern    char   *SfLabel          ( SpecFile *sf, long index, long column,
                                              int *error );

   /*
    * Motors
    */
DllExport extern  long    SfAllMotors        ( SpecFile *sf, long index,
                                                char ***names, int *error );
DllExport extern  char  * SfMotor            ( SpecFile *sf, long index,
                                                long number, int *error );
DllExport extern  long    SfAllMotorPos      ( SpecFile *sf, long index,
                                                double **pos, int *error );
DllExport extern  double  SfMotorPos         ( SpecFile *sf, long index,
                                                long number, int *error );
DllExport extern  double  SfMotorPosByName   ( SpecFile *sf, long index,
                                                char *name, int *error );

   /*
    * Data
    */
DllExport extern  long  SfNoDataLines ( SpecFile *sf, long index, int *error );
DllExport extern  int   SfData        ( SpecFile *sf, long index,
                                double ***data, long **data_info, int *error );
DllExport extern  long  SfDataAsString ( SpecFile *sf, long index,
                                   char ***data, int *error );
DllExport extern  long  SfDataLine      ( SpecFile *sf, long index, long line,
                                             double **data_line, int *error );
DllExport extern  long  SfDataCol       ( SpecFile *sf, long index, long col,
                                             double **data_col, int *error );
DllExport extern  long  SfDataColByName ( SpecFile *sf, long index,
                                  char *label, double **data_col, int *error );

  /*
   * MCA functions
   */
DllExport extern long SfNoMca   ( SpecFile *sf, long index, int *error );
DllExport extern int  SfGetMca  ( SpecFile *sf, long index, long mcano,
                                          double **retdata, int *error );
DllExport extern long SfMcaCalib ( SpecFile *sf, long index, double **calib,
                                          int *error );

  /*
   * Write and write related functions
   */
DllExport extern  SpecFileOut  *SfoInit  ( SpecFile *sf, int *error );
DllExport extern  void      SfoClose     ( SpecFileOut *sfo );
DllExport extern  long      SfoSelectAll ( SpecFileOut *sfo, int *error );
DllExport extern  long      SfoSelectOne ( SpecFileOut *sfo, long index,
                                                int *error );
DllExport extern  long      SfoSelect    ( SpecFileOut *sfo, long *list,
                                                int *error );
DllExport extern  long      SfoSelectRange ( SpecFileOut *sfo, long begin,
                                                long end, int *error );
DllExport extern  long      SfoRemoveOne ( SpecFileOut *sfo, long index,
                                                int *error );
DllExport extern  long      SfoRemove    ( SpecFileOut *sfo, long *list,
                                                int *error );
DllExport extern  long      SfoRemoveRange ( SpecFileOut *sfo, long begin,
                                                long end, int *error );
DllExport extern  long      SfoRemoveAll ( SpecFileOut *sfo, int *error );
DllExport extern  long      SfoWrite     ( SpecFileOut *sfo, char *name,
                                                int *error );
DllExport extern  long      SfoGetList   ( SpecFileOut *sfo, long **list,
                                                int *error );
  /*
   * Memory free functions
   */
DllExport extern  void      freeArrNZ    ( void ***ptr, long no_lines );
DllExport extern  void      freePtr      ( void *ptr );

  /*
   * Sf Tools
   */
DllExport extern  void      SfShow       ( SpecFile *sf );
DllExport extern  void      SfShowScan   ( SpecFile *sf ,long index);
  /*
   * Error
   */
DllExport extern  char     *SfError        ( int code );

#endif  /*  SPECFILE_H  */
