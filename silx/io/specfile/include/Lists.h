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
/***************************************************************************
 *
 *  File:            Lists.h
 *
 *  Description:     Include file for dealing with lists.
 *
 *  Author:          Vicente Rey
 *
 *  Created:         22 May 1995
 *
 *    (copyright by E.S.R.F.  March 1995)
 *
 ***************************************************************************/
#ifndef LISTS_H
#define LISTS_H

/* #include <malloc.h> */

typedef struct _ObjectList {
  struct _ObjectList   *next;
  struct _ObjectList   *prev;
  void                 *contents;
} ObjectList;

typedef struct _ListHeader {
  struct _ObjectList   *first;
  struct _ObjectList   *last;
} ListHeader;

extern  ObjectList * findInList     ( ListHeader *list, int (*proc)(void *,void *), void *value );
extern  long         addToList      ( ListHeader *list, void *object,long size);
extern  void         unlinkFromList ( ListHeader *list,  ObjectList *element);

#endif  /*  LISTS_H  */
