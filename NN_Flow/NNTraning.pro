#-------------------------------------------------
#
# Project created by QtCreator 2015-05-19T00:27:39
#
#-------------------------------------------------
QMAKE_CXXFLAGS += -std=c++11
#QMAKE_CXXFLAGS += -fopenmp
#QMAKE_LFLAGS += -fopenmp
QT       += core gui

#QMAKE_LFLAGS += -Wall

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NNTraning
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    netlayer.cpp \
    nndetect.cpp

HEADERS  += mainwindow.h \
    netlayer.h \
    nndetect.h

FORMS    += mainwindow.ui


# remove possible other optimization flags
QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

# add the desired -O3 if not present
QMAKE_CXXFLAGS_RELEASE *= -O3
QMAKE_CXXFLAGS_RELEASE *= -Ofast

