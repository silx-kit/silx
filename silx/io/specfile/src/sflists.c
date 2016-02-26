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
 *   File:          sflists.c
 *
 *   Project:       SpecFile library
 *
 *   Description:   Functions to handle lists
 *
 *   Author:        V.Rey
 *
 *   Date:          $Date: 2003/03/06 17:00:42 $
 *
 ************************************************************************/
/*
 *   Log: $Log: sflists.c,v $
 *   Log: Revision 1.1  2003/03/06 17:00:42  sole
 *   Log: Initial revision
 *   Log:
 *   Log: Revision 3.0  2000/12/20 14:17:19  rey
 *   Log: Python version available
 *   Log:
 * Revision 2.1  2000/07/31  19:03:25  19:03:25  rey (Vicente Rey-Bakaikoa)
 * SfUpdate and bug corrected in ReadIndex
 *
 * Revision 2.0  2000/04/13  13:28:54  13:28:54  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Lists.h>

/*
 * Function declaration
 */
ObjectList * findInList     ( ListHeader *list, int (*proc)(void *,void *), void *value );
long         addToList      ( ListHeader *list, void *object, long size );
void         unlinkFromList ( ListHeader *list, ObjectList *element );

static long  linkToList     ( ListHeader *list, void *object );


/*********************************************************************
 *   Function:		ObjectList *findInList( list, proc, value )
 *
 *   Description:	Looks for an list element.
 *
 *   Parameters:
 *		Input :	(1) ListHeader pointer
 *			(2) Comp. procedure
 *			(3) value
 *   Returns:
 *		Pointer to the found element ,
 *		NULL if not found .
 *
 *********************************************************************/
ObjectList *
findInList( ListHeader *list, int (*proc)(void * , void *), void *value )
{
     register ObjectList	*ptr;

     for ( ptr=list->first ; ptr ; ptr=ptr->next ) {
          if ( (*proc)(ptr->contents, value) ) {
	       return( ptr );
	  }
     }
     return (ObjectList *)NULL;
}


/*********************************************************************
 *   Function:		int addToList( list, object, size )
 *
 *   Description:	Adds an element to the list.
 *
 *   Parameters:
 *		Input :	(1) List pointer
 *			(2) Pointer to the new element
 *			(3) Size of the new element
 *   Returns:
 *		(  0 ) => OK
 *	        ( -1 ) => error
 *
 *********************************************************************/
long
addToList( ListHeader *list, void *object, long size )
{
     void    *newobj;

     if ( (newobj = (void *)malloc(size)) == (void *)NULL ) return( -1 );
     memcpy(newobj, object, size);

     return( linkToList( list, newobj ) );

}


/*********************************************************************
 *   Function:		int linkToList( list, object )
 *
 *   Description:	Adds an element to the list.
 *
 *   Parameters:
 *		Input:	(1) ListHeader pointer
 *			(2) pointer to the new element
 *   Returns:
 *		(  0 ) => OK
 *	        ( -1 ) => error
 *
 *********************************************************************/
static long
linkToList( ListHeader *list, void *object )
{
     ObjectList	*newobj;


     if ((newobj = (ObjectList *) malloc(sizeof(ObjectList))) ==
	    (ObjectList *) NULL)  return( -1 );

     newobj->contents	= object;
     newobj->prev	= list->last;
     newobj->next	= NULL;

     if (list->first == (ObjectList *)NULL) {
         list->first  = newobj;
     } else {
         (list->last)->next = newobj;
     }

     list->last = newobj;
     return( 0 );
}


/*********************************************************************
 *   Function:		int unlinkFromList( list, element )
 *
 *   Description:	Removes an element from the list.
 *
 *   Parameters:
 *		Input :	(1) List pointer
 *			(2) Pointer to the element
 *
 *********************************************************************/
void
unlinkFromList( ListHeader *list, ObjectList *element )
{

     if ( element != (ObjectList *)NULL ) {
	  if ( element->next != (ObjectList *)NULL ) {
	       element->next->prev = element->prev;
	  }
	  else {
	       list->last = element->prev ;
	  }
	  if ( element->prev != (ObjectList *)NULL ) {
	       element->prev->next = element->next;
	  }
	  else {
	       list->first = element->next;
	  }
	  free( element->contents );
	  free( element );
     }
}

