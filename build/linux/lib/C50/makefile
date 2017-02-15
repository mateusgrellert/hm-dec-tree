# the SOURCE definiton lets move your makefile to another position
CONFIG 			= LIBRARY

# set pathes to the correct directories
SRC_DIR			= ../../../../source/Lib/C50
INC_DIR			= ../../../../source/Lib
LIB_DIR			= ../../../../lib
BIN_DIR			= ../../../../bin

SRC_DIR1		= ../../../../source/Lib/libmd5
SRC_DIR2		=
SRC_DIR3		=
SRC_DIR4		=


USER_INC_DIRS	= -I$(SRC_DIR)
USER_LIB_DIRS	=

ifeq ($(HIGHBITDEPTH), 1)
HBD=HighBitDepth
else
HBD=
endif

# intermediate directory for object files
OBJ_DIR			= ./objects$(HBD)

# the library name
PRJ_NAME	 	= TLibCommon$(HBD)

# version information
MAJOR_VER		= 0
MINOR_VER		= 1
VER				= $(MAJOR_VER).$(MINOR_VER)

# defines to set
DEFS			= -DMSYS_LINUX

# set objects

OBJS          	= \
			$(OBJ_DIR)/utility.o \
			$(OBJ_DIR)/update.o \
			$(OBJ_DIR)/trees.o \
			$(OBJ_DIR)/subset.o \
			$(OBJ_DIR)/modelfiles.o \
			$(OBJ_DIR)/info.o \
			$(OBJ_DIR)/hooks.o \
			$(OBJ_DIR)/global.o \
			$(OBJ_DIR)/getnames.o \
			$(OBJ_DIR)/getdata.o \
			$(OBJ_DIR)/formtree.o \
			$(OBJ_DIR)/discr.o \
			$(OBJ_DIR)/contin.o \
			$(OBJ_DIR)/construct.o \
			$(OBJ_DIR)/attwinnow.o \

LIBS				= -lpthread

DEBUG_LIBS			=
RELEASE_LIBS		=

STAT_LIBS			=
DYN_LIBS			= -ldl

# the libraries to link with
STAT_DEBUG_LIBS		=
STAT_RELEASE_LIBS	=
DYN_DEBUG_LIBS		=
DYN_RELEASE_LIBS	=

# name of the base makefile
MAKE_FILE_NAME		= ../../common/makefile.base

# include the base makefile
include $(MAKE_FILE_NAME)
