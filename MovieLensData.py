
from collections import defaultdict

import numpy as np

import os
import re
import sys
import csv

from surprise import Reader
from surprise import Dataset

class MovieLensData:

    idToName = {}
    nameToID = {}

    moviesData = 'ml-latest-small/movies.csv'
    ratingsData = 'ml-latest-small/ratings.csv'
 
    def loadMovieLens(self):
        
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        self.nameToID = {}
        ratings_Dataset = 0
        self.idToName = {}
    
        ratings_Dataset = Dataset.load_from_file(self.ratingsData, reader=reader)

        with open(self.moviesData, newline='', encoding='ISO-8859-1') as csvfile:
                reader_movie = csv.reader(csvfile)
                next(reader_movie)  
                for row in reader_movie:

                    name = row[1]
                    ID = int(row[0])

                    self.nameToID[name] = ID
                    self.idToName[ID] = name
                   
        return ratings_Dataset

    def getUserRatings(self, required_user):
        
        hit = False
        ratings_user = []
        
        with open(self.ratingsData, newline='') as csvfile:
            reader_rating = csv.reader(csvfile)
            next(reader_rating)
            for row in reader_rating:
                currentUserID = int(row[0])
                if (required_user == currentUserID):
                    
                    rating = float(row[2])
                    ID = int(row[1])
                    ratings_user.append((ID, rating))
                    hit = True

                if ((required_user != currentUserID) and hit):
                    break

        return ratings_user

    def computePopularity(self):

        rankings_dict = defaultdict(int)
        ratings_dict = defaultdict(int)
        
        with open(self.ratingsData, newline='') as csvfile:
            reader_rating = csv.reader(csvfile)
            next(reader_rating)
            for row in reader_rating:
                ID = int(row[1])
                ratings_dict[ID] += 1
        rank = 1
        for ID, ratingCount in sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True):
            rankings_dict[ID] = rank
            rank += 1
        return rankings_dict
    
    def returnGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesData, newline='', encoding='ISO-8859-1') as csvfile:
            reader_movie = csv.reader(csvfile)
            next(reader_movie)
            for row in reader_movie:
                ID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[ID] = genreIDList

        for (ID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[ID] = bitfield            
        
        return genres
    
    def returnYears(self):

        years_dict = defaultdict(int)
        pattern = re.compile(r"(?:\((\d{4})\))?\s*$")
        
        with open(self.moviesData, newline='', encoding='ISO-8859-1') as csvfile:
            reader_movie = csv.reader(csvfile)
            next(reader_movie)
            for row in reader_movie:
                title = row[1]
                ID = int(row[0])
                mov = pattern.search(title)
                year = mov.group(1)
                if year:
                    years_dict[ID] = int(year)
        return years_dict
    
    def returnMES(self):
        mes = defaultdict(list)
        with open("Features_MES.csv", newline='') as csvfile:
            reader_mes = csv.reader(csvfile)
            next(reader_mes)
            for row in reader_mes:
                ID = int(row[0])
                shotLength = float(row[1])
                colourVariance = float(row[2])
                motion = float(row[4])
                lighting = float(row[6])
                shots = float(row[7])
                mes[ID] = [motion,shots,lighting,colourVariance,shotLength]

        return mes
    
    def returnMovieName(self, ID):
        if ID in self.idToName:
            return self.idToName[ID]
        else:
            return "Unknown Name"