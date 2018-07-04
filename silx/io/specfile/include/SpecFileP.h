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
 *  File:            SpecFileP.h
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
#ifndef SPECFILE_P_H
#define SPECFILE_P_H

/*
 * Defines.
 */
#define   FILE_HEADER         0
#define   SCAN                1

#define   FROM_SCAN           0
#define   FROM_FILE           1

#define   SF_COMMENT         'C'
#define   SF_DATE            'D'
#define   SF_EPOCH           'E'
#define   SF_FILE_NAME       'F'
#define   SF_GEOMETRY        'G'
#define   SF_INTENSITY       'I'
#define   SF_LABEL           'L'
#define   SF_MON_NORM        'M'
#define   SF_COLUMNS         'N'
#define   SF_MOTOR_NAMES     'O'
#define   SF_MOTOR_POSITIONS 'P'
#define   SF_RECIP_SPACE     'Q'
#define   SF_RESULTS         'R'
#define   SF_SCAN_NUM        'S'
#define   SF_TIME_NORM       'T'
#define   SF_USER_DEFINED    'U'
#define   SF_TEMPERATURE     'X'
#define   SF_MCA_DATA        '@'

/*
 * Library internal functions
 */
extern  int        sfSetCurrent    ( SpecFile *sf,   long index, int *error);
extern ObjectList *findScanByIndex ( ListHeader *list, long index );
extern ObjectList *findScanByNo    ( ListHeader *list, long scan_no, long order );
extern void        freeArr         ( void ***ptr, long lines );
extern void        freeAllData     ( SpecFile *sf );
extern long        mulstrtod       ( char *str, double **arr, int *error );
extern int         sfGetHeaderLine ( SpecFile *sf, int from, char character,
                                             char **buf,int *error);

#endif  /*  SPECFILE_P_H  */
