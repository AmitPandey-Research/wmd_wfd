{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9a7f654f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "this is a test python notebook to be run using bash file \n",
      "\n",
      "processing the i = 4 \n",
      "processing the i = 2 \n",
      "processing the i = 6 \n",
      "processing the i = 0 \n",
      "processing the i = 12 \n",
      "processing the i = 14 \n",
      "processing the i = 16 \n",
      "processing the i = 18 \n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "processing the i = 5 \n",
      "\n",
      "processing the i = 3 \n",
      "processing the i = 7 \n",
      "\n",
      "processing the i = 13 \n",
      "\n",
      "processing the i = 15 \n",
      "processing the i = 19 \n",
      "processing the i = 17 \n",
      "processing the i = 1 \n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "processing the i = 20 \n",
      "processing the i = 22 \n",
      "processing the i = 24 \n",
      "processing the i = 28 \n",
      "\n",
      "processing the i = 26 \n",
      "processing the i = 30 \n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "processing the i = 23 \n",
      "processing the i = 21 \n",
      "processing the i = 27 \n",
      "processing the i = 25 \n",
      "\n",
      "processing the i = 32 \n",
      "\n",
      "\n",
      "\n",
      "\n",
      "processing the i = 34 \n",
      "\n",
      "processing the i = 36 \n",
      "processing the i = 40 \n",
      "processing the i = 33 \n",
      "processing the i = 38 \n",
      "\n",
      "\n",
      "\n",
      "processing the i = 37 \n",
      "\n",
      "processing the i = 41 \n",
      "processing the i = 39 \n",
      "\n",
      "processing the i = 42 \n",
      "\n",
      "\n",
      "\n",
      "processing the i = 35 \n",
      "processing the i = 44 \n",
      "\n",
      "processing the i = 46 \n",
      "\n",
      "\n",
      "processing the i = 48 \n",
      "\n",
      "\n",
      "processing the i = 45 \n",
      "processing the i = 47 \n",
      "\n",
      "processing the i = 43 \n",
      "\n",
      "processing the i = 49 \n",
      "\n",
      "\n",
      "\n",
      "processing the i = 31 \n",
      "\n",
      "processing the i = 29 \n",
      "\n",
      "processing the i = 8 \n",
      "processing the i = 10 \n",
      "\n",
      "\n",
      "processing the i = 9 \n",
      "processing the i = 11 \n",
      "\n",
      "\n",
      " printing the output of map function :  [{0: array([], dtype=int64), 1: array([0])}, {0: array([], dtype=int64), 1: array([0])}, {2: array([0, 1]), 3: array([0, 1, 2])}, {2: array([0, 1]), 3: array([0, 1, 2])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4])}, {6: array([0, 1, 2, 3, 4, 5]), 7: array([0, 1, 2, 3, 4, 5, 6])}, {6: array([0, 1, 2, 3, 4, 5]), 7: array([0, 1, 2, 3, 4, 5, 6])}, {8: array([0, 1, 2, 3, 4, 5, 6, 7]), 9: array([0, 1, 2, 3, 4, 5, 6, 7, 8])}, {8: array([0, 1, 2, 3, 4, 5, 6, 7]), 9: array([0, 1, 2, 3, 4, 5, 6, 7, 8])}, {10: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 11: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])}, {10: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 11: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])}, {16: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), 17: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])}, {16: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), 17: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])}, {18: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17]), 19: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18])}, {18: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17]), 19: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), 22: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21]), 23: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), 22: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21]), 23: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26])}, {18: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17]), 19: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18]), 28: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 29: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])}, {18: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17]), 19: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18]), 28: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 29: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])}, {16: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), 17: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]), 30: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]), 31: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])}, {16: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), 17: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]), 30: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]), 31: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4]), 32: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 33: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4]), 32: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 33: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), 22: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21]), 23: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22]), 34: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]), 35: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34])}, {14: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 15: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), 22: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21]), 23: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22]), 34: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]), 35: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24]), 36: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35]), 37: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24]), 36: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35]), 37: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26]), 38: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37]), 39: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26]), 38: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37]), 39: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20]), 40: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39]), 41: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20]), 40: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39]), 41: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24]), 36: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35]), 37: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36]), 42: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41]), 43: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42])}, {2: array([0, 1]), 3: array([0, 1, 2]), 24: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23]), 25: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24]), 36: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35]), 37: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36]), 42: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41]), 43: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20]), 40: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39]), 41: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40]), 44: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43]), 45: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])}, {12: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), 13: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), 20: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19]), 21: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20]), 40: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39]), 41: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40]), 44: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43]), 45: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26]), 38: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37]), 39: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38]), 46: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]), 47: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])}, {0: array([], dtype=int64), 1: array([0]), 26: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25]), 27: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26]), 38: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37]), 39: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38]), 46: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]), 47: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4]), 32: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 33: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]), 48: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]), 49: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])}, {4: array([0, 1, 2, 3]), 5: array([0, 1, 2, 3, 4]), 32: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 33: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]), 48: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]), 49: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
      "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
      "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])}] \n",
      "\n",
      "time taken is :  0.16489672660827637\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'dic' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-a63d8b9258ea>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     32\u001b[0m \u001b[0ma_file\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"testdic.json\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"w\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 33\u001b[0;31m \u001b[0mjson\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdump\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdic\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ma_file\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     34\u001b[0m \u001b[0ma_file\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     35\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'dic' is not defined"
     ]
    }
   ],
   "source": [
    "print(\"this is a test python notebook to be run using bash file \\n\")\n",
    "\n",
    "import numpy as np\n",
    "from multiprocessing import Pool\n",
    "import json\n",
    "import time\n",
    "\n",
    "st = time.time()\n",
    "\n",
    "dicti = {}\n",
    "\n",
    "def fun(i):\n",
    "    print(f\"processing the i = {i} \\n\")\n",
    "    dicti[i] = np.arange(i)\n",
    "    \n",
    "    return dicti\n",
    "    \n",
    "    \n",
    "with Pool(10) as p:\n",
    "      a = p.map(fun,np.arange(50))\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "print(\" printing the output of map function : \",a ,\"\\n\")\n",
    "\n",
    "\n",
    "et = time.time()\n",
    "\n",
    "print(\"time taken is : \", et-st)\n",
    "\n",
    "a_file = open(\"testdic.json\", \"w\")\n",
    "json.dump(dic, a_file)\n",
    "a_file.close()\n",
    "\n",
    "a_file = open(\"testdic.json\", \"r\")\n",
    "output = a_file.read()\n",
    "print(output)\n",
    "\n",
    "a_file.close()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32cc71c9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
