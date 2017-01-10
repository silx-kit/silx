#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
 *   File:          sflabel.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Access to labels and motors
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2003/02/03 13:15:35 $
 *
 ************************************************************************/
/*
 *   Log:
 * $Log: sflabel.c,v $
 * Revision 1.3  2003/02/03 13:15:35  rey
 * Small change in handling of empty spaces at the beginning of the label buffer
 *
 * Revision 1.2  2002/11/20 09:56:31  sole
 * Some macros leave more than 1 space between #L and the first label.
 * Routine modified to be able to deal with already collected data.
 * The offending macro(s) should be re-written.
 *
 * Revision 1.1  2002/11/20 08:21:34  sole
 * Initial revision
 *
 * Revision 3.0  2000/12/20 14:17:19  rey
 * Python version available
 *
 * Revision 2.2  2000/12/20 12:12:08  rey
 * bug corrected with SfAllMotors
 *
 * Revision 2.1  2000/07/31  19:05:10  19:05:10  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
#include <SpecFile.h>
#include <SpecFileP.h>
#include <locale_management.h>
#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
#include <locale.h>
#endif
#endif

/*
 * Declarations
 */
DllExport char * SfLabel         ( SpecFile *sf, long index, long column,
                                      int *error );
DllExport long   SfAllLabels     ( SpecFile *sf, long index, char ***labels,
                                      int *error );
DllExport char * SfMotor         ( SpecFile *sf, long index, long number,
                                      int *error );
DllExport long   SfAllMotors     ( SpecFile *sf, long index, char ***names,
                                      int *error );
DllExport double SfMotorPos      ( SpecFile *sf, long index, long number,
                                      int *error );
DllExport double SfMotorPosByName( SpecFile *sf, long index, char *name,
                                      int *error );
DllExport long   SfAllMotorPos   ( SpecFile *sf, long index, double **pos,
                                      int *error );


/*********************************************************************
 *   Function:		char *SfLabel( sf, index, column, error )
 *
 *   Description:	Reads one label.
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) Scan index
 *			(3) Column number
 *		Output:	(4) Error number
 *   Returns:
 *			Pointer to the label ,
 *			or NULL if errors occured.
 *   Possible errors:
 *			SF_ERR_MEMORY_ALLOC	| => getStrFromArr()
 *			SF_ERR_LABEL_NOT_FOUND
 *			SF_ERR_LINE_EMPTY	|
 *			SF_ERR_LINE_NOT_FOUND	|
 *			SF_ERR_SCAN_NOT_FOUND	| => SfAllLabels()
 *			SF_ERR_FILE_READ	|
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport char *
SfLabel( SpecFile *sf, long index, long column, int *error )
{

     char   **labels=NULL;
     long     no_labels;
     char    *label=NULL;
     long     selection;

     if (sfSetCurrent(sf,index,error) == -1)
          return((char *)NULL);

     if (sf->no_labels != -1 ) {
        no_labels = sf->no_labels;
     } else {
        no_labels = SfAllLabels(sf,index,&labels,error);
     }

     if (no_labels == 0 || no_labels == -1)  return((char *)NULL);

     if ( column < 0 ) {
           selection = no_labels + column;
     } else {
           selection = column - 1;
     }

     if (selection < 0 || selection > no_labels - 1 ) {
         *error = SF_ERR_COL_NOT_FOUND;
          if (labels != (char **) NULL )
              freeArrNZ((void ***)&labels,no_labels);
          return((char *)NULL);
     }

     if (labels != (char **)NULL) {
         label = (char *)strdup(labels[selection]);
         freeArrNZ((void ***)&labels,no_labels);
     } else {
         label = (char *) strdup(sf->labels[selection]);
     }
     return( label );
}


/*********************************************************************
 *   Function:		long SfAllLabels( sf, index, labels, error )
 *
 *   Description:	Reads all labels in #L lines
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) Scan index
 *		Output:	(3) Labels
 *			(4) Error number
 *   Returns:
 *			Number of labels
 *			( -1 ) if error.
 *   Possible errors:
 *			SF_ERR_MEMORY_ALLOC	||=> cpyStrArr(),lines2words()
 *			SF_ERR_SCAN_NOT_FOUND	| => SfHeader()
 *			SF_ERR_FILE_READ	|
 *			SF_ERR_LINE_EMPTY
 *			SF_ERR_LINE_NOT_FOUND
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long
SfAllLabels( SpecFile *sf, long index, char ***labels, int *error )
{
     static char tmplab[40];

     char **labarr;
     char  *onelabel;

     char  *ptr,
           *buf=NULL;

     long      no_labels = 0;
     short     i;

    /*
     * select scan
     */
     if (sfSetCurrent(sf,index,error) == -1) {
         *labels = NULL;
          return(0);
     }

    /*
     * Do not do it if already done
     */
     if (sf->labels != (char **)NULL ) {
         labarr = (char **)malloc(sizeof(char *) * sf->no_labels);
         for ( i=0;i<sf->no_labels;i++)
             labarr[i] = (char *)strdup(sf->labels[i]);
        *labels = labarr;
         return(sf->no_labels);
     }

    /*
     * else..
     */
     if (sfGetHeaderLine(sf,FROM_SCAN,SF_LABEL,&buf,error) == -1) {
         *labels = NULL;
          return(0);
     }

     if ( buf[0] == '\0') {
        *labels = NULL;
         return(0);
     }

     if ( (labarr = (char **)malloc( sizeof(char *))) == (char **)NULL) {
         *error = SF_ERR_MEMORY_ALLOC;
          return(-1);
     }

     no_labels = 0;
     i = 0;

     /*
      * avoid problem of having too many spaces at the beginning
      * with bad written macros -> added check for empty string
      *
      *   get rid of spaces at the beginning of the string buffer
      */

     ptr = buf;
     while((ptr < buf + strlen(buf) -1) && (*ptr == ' ')) ptr++;

     for (i=0;ptr < buf + strlen(buf) -1;ptr++,i++) {
         if (*ptr==' ' && *(ptr+1) == ' ') { /* two spaces delimits one label */
             tmplab[i] = '\0';

             labarr = (char **)realloc( labarr, (no_labels+1) * sizeof(char *));
             onelabel = (char *) malloc (i+2);
             strcpy(onelabel,tmplab);
             labarr[no_labels]  = onelabel;

             no_labels++;
             i=-1;
             for(;*(ptr+1) == ' ' && ptr < buf+strlen(buf)-1;ptr++);
         } else {
             tmplab[i] = *ptr;
         }
     }

     if (*ptr != ' ') {
        tmplab[i]   = *ptr;
        i++;
     }
     tmplab[i] = '\0';

     labarr = (char **)realloc( labarr, (no_labels+1) * sizeof(char *));
     onelabel = (char *) malloc (i+2);
     strcpy(onelabel,tmplab);
     labarr[no_labels]  = onelabel;

     no_labels++;

    /*
     * Save in specfile structure
     */
     sf->no_labels = no_labels;
     sf->labels    = (char **) malloc( sizeof(char *) * no_labels);
     for (i=0;i<no_labels;i++)
           sf->labels[i] = (char *) strdup(labarr[i]);

    *labels = labarr;
     return( no_labels );
}


/*********************************************************************
 *   Function:		long SfAllMotors( sf, index, names, error )
 *
 *   Description:	Reads all motor names in #O lines (in file header)
 *
 *   Parameters:
 *		Input :	(1) SpecScan pointer
 *			(2) Scan index
 *		Output:	(3) Names
 *			(4) Error number
 *   Returns:
 *			Number of found names
 *			( -1 ) if errors.
 *   Possible errors:
 *			SF_ERR_SCAN_NOT_FOUND
 *			SF_ERR_LINE_NOT_FOUND
 *			SF_ERR_LINE_EMPTY
 *			SF_ERR_MEMORY_ALLOC    || => cpyStrArr(),lines2words()
 *			SF_ERR_FILE_READ	|
 *			SF_ERR_HEADER_NOT_FOUND	| => SfFileHeader()
 *
 *   Remark:  The memory allocated should be freed by the application
 *
 *********************************************************************/
DllExport long
SfAllMotors( SpecFile *sf, long index, char ***names, int *error )
{
     char **lines;
     char  *thisline,
           *endline;

     char **motarr;
     char  *onemot;

     static char tmpmot[40];

     char  *ptr;

     long      motct = 0;
     long      no_lines;
     short     i,j;

    /*
     * go to scan
     */
     if (sfSetCurrent(sf,index,error) == -1) {
         *names = NULL;
          return(0);
     }

    /*
     * if motor names for this scan have already been read
     */
     if (sf->motor_names != (char **)NULL) {
         motarr = (char **)malloc(sizeof(char *) * sf->no_motor_names);
         for (i=0;i<sf->no_motor_names;i++) {
             motarr[i] = (char *) strdup (sf->motor_names[i]);
         }
        *names = motarr;
         return(sf->no_motor_names);
     }

    /*
     * else
     */
     no_lines =  SfHeader(sf, index,"O",&lines,error);
     if (no_lines == -1 || no_lines == 0 ) {
         *names = (char **) NULL;
          return(-1);
     }

     if ( (motarr = (char **)malloc( sizeof(char *))) == (char **)NULL) {
         *error = SF_ERR_MEMORY_ALLOC;
          return(-1);
     }

     motct = 0;

     for (j=0;j<no_lines;j++) {
         thisline = lines[j] + 4;
         endline  = thisline + strlen(thisline);
         for(ptr=thisline;*ptr == ' ';ptr++);
         for (i=0;ptr < endline -2;ptr++,i++) {
            if (*ptr==' ' && *(ptr+1) == ' ') {
               tmpmot[i] = '\0';

               motarr = (char **)realloc( motarr, (motct+1) * sizeof(char *));
               onemot = (char *) malloc (i+2);
               strcpy(onemot,tmpmot);
               motarr[motct]  = onemot;

               motct++;
               i=-1;
               for(;*(ptr+1) == ' ' && ptr < endline -1;ptr++);
            } else {
               tmpmot[i] = *ptr;
            }
        }
        if (*ptr != ' ') { tmpmot[i]   = *ptr; i++; }
        ptr++;
        if (*ptr != ' ') { tmpmot[i]   = *ptr; i++; }

        tmpmot[i] = '\0';
        motarr = (char **)realloc( motarr, (motct+1) * sizeof(char *));

        onemot = (char *) malloc (i+2);
        strcpy(onemot,tmpmot);
        motarr[motct]  = onemot;

        motct++;

   }

  /*
   * Save in specfile structure
   */
   sf->no_motor_names = motct;
   sf->motor_names = (char **)malloc(sizeof(char *) * motct);
   for (i=0;i<motct;i++) {
        sf->motor_names[i] = (char *)strdup(motarr[i]);
   }

  *names = motarr;
   return( motct );

}


DllExport char *
SfMotor( SpecFile *sf, long index, long motnum, int *error )
{

     char   **motors=NULL;
     long     nb_mot;
     char    *motor=NULL;
     long     selection;

    /*
     * go to scan
     */
     if (sfSetCurrent(sf,index,error) == -1) {
          return((char *)NULL);
     }

     if ( sf->no_motor_names != -1 ) {
        nb_mot = sf->no_motor_names;
     } else {
        nb_mot = SfAllMotors(sf,index,&motors,error);
     }

     if (nb_mot == 0 || nb_mot == -1)  return((char *)NULL);

     if ( motnum < 0 ) {
           selection = nb_mot + motnum;
     } else {
           selection = motnum - 1;
     }

     if (selection < 0 || selection > nb_mot - 1 ) {
         *error = SF_ERR_COL_NOT_FOUND;
          if (motors != (char **) NULL)
              freeArrNZ((void ***)&motors,nb_mot);
          return((char *)NULL);
     }

     if (motors != (char **) NULL) {
         motor = (char *)strdup(motors[selection]);
         freeArrNZ((void ***)&motors,nb_mot);
     } else {
         motor = (char *)strdup(sf->motor_names[selection]);
     }
     return( motor );
}


DllExport long
SfAllMotorPos ( SpecFile *sf, long index, double **retpos, int *error )
{
     char **lines;
     char  *thisline,
           *endline;

     double *posarr;

     static double pos[200];
     static char   posstr[40];

     char  *ptr;

     long      motct = 0;
     long      no_lines;
     short     i,j;

#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
	char *currentLocaleBuffer;
	char localeBuffer[21];
#endif
#endif

     if (sfSetCurrent(sf,index,error) == -1) {
         *retpos = (double *) NULL;
          return(0);
     }

    /*
     * if motors position for this scan have already been read
     */
     if (sf->motor_pos != (double *)NULL) {
         posarr = (double *)malloc(sizeof(double) * sf->no_motor_pos);
         for (i=0;i<sf->no_motor_pos;i++) {
             posarr[i] = sf->motor_pos[i];
         }
        *retpos = posarr;
         return(sf->no_motor_pos);
     }

    /*
     * else
     */
     no_lines =  SfHeader(sf, index,"P",&lines,error);

     if (no_lines == -1 || no_lines == 0 ) {
         *retpos = (double *) NULL;
          return(-1);
     }

     motct = 0;
#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
	currentLocaleBuffer = setlocale(LC_NUMERIC, NULL);
	strcpy(localeBuffer, currentLocaleBuffer);
	setlocale(LC_NUMERIC, "C\0");
#endif
#endif
     for (j=0;j<no_lines;j++) {
         thisline = lines[j] + 4;
         endline  = thisline + strlen(thisline);
         for(ptr=thisline;*ptr == ' ';ptr++);
         for (i=0;ptr < endline -1;ptr++,i++) {
            if (*ptr==' ') {
               posstr[i] = '\0';

               pos[motct]  = PyMcaAtof(posstr);

               motct++;
               i=-1;
               for(;*(ptr+1) == ' ' && ptr < endline -1;ptr++);
            } else {
               posstr[i] = *ptr;
            }
         }
         if (*ptr != ' ') {
            posstr[i]   = *ptr;
            i++;
         }
         posstr[i]   = '\0';
         pos[motct]  = PyMcaAtof(posstr);

         motct++;

	 }

#ifndef _GNU_SOURCE
#ifdef PYMCA_POSIX
	setlocale(LC_NUMERIC, localeBuffer);
#endif
#endif

     /*
      * Save in specfile structure
      */
      sf->no_motor_pos = motct;
      sf->motor_pos    = (double *)malloc(sizeof(double) * motct);
      memcpy(sf->motor_pos,pos,motct * sizeof(double));

     /*
      * and return
      */
      posarr = (double *) malloc ( sizeof(double) * motct ) ;
      memcpy(posarr,pos,motct * sizeof(double));

     *retpos = posarr;

      return( motct );
}


DllExport double
SfMotorPos( SpecFile *sf, long index, long motnum, int *error )
{

     double  *motorpos=NULL;
     long     nb_mot;
     double   retpos;
     long     selection;

     if (sfSetCurrent(sf,index,error) == -1)
          return(HUGE_VAL);

     if (sf->no_motor_pos != -1 ) {
       nb_mot = sf->no_motor_pos;
     } else {
       nb_mot = SfAllMotorPos(sf,index,&motorpos,error);
     }

     if (nb_mot == 0 || nb_mot == -1)  return(HUGE_VAL);

     if ( motnum < 0 ) {
           selection = nb_mot + motnum;
     } else {
           selection = motnum - 1;
     }

     if (selection < 0 || selection > nb_mot - 1 ) {
         *error = SF_ERR_COL_NOT_FOUND;
          if (motorpos != (double *)NULL)
              free(motorpos);
          return(HUGE_VAL);
     }

     if (motorpos != (double *)NULL) {
         retpos = motorpos[selection];
         free(motorpos);
     } else {
         retpos = sf->motor_pos[selection];
     }
     return( retpos );
}


DllExport double
SfMotorPosByName( SpecFile *sf, long index, char *name, int *error )
{
     char **motors=NULL;

     long  nb_mot,
           idx,
           selection;
     short tofree=0;

     if (sfSetCurrent(sf,index,error) == -1)
          return(HUGE_VAL);

     if ( sf->no_motor_names != -1 ) {
        nb_mot = sf->no_motor_names;
        motors = sf->motor_names;
     } else {
        nb_mot = SfAllMotors(sf,index,&motors,error);
        tofree=1;
     }

     if (nb_mot == 0 || nb_mot == -1)  return(HUGE_VAL);

     for (idx = 0;idx<nb_mot;idx++) {
         if (!strcmp(name,motors[idx])) break;
     }

     if (idx == nb_mot) {
           if (tofree) freeArrNZ((void ***)&motors,nb_mot);
          *error = SF_ERR_MOTOR_NOT_FOUND;
           return(HUGE_VAL);
     }

     selection = idx+1;

     return(SfMotorPos(sf,index,selection,error));
}
