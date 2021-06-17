import spacy
import pandas as pd
import numpy as np
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
import json
import os
from datetime import datetime
import dateutil.parser as parser
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import copy
import io
import math
from io import StringIO
import boto3
import pickle
from collections import Counter
import re
import sys
import time

stops = ['i', 're:', 'fwd:', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table('DEV-INTERESTS')
primary_key = 'pk'

class predefinedFunctions:

	def __init__(self):
		self.scaler = MinMaxScaler(feature_range=(0.1, 1))

	def normalize(self, df):
		return self.scaler.fit_transform(df)


	def groupbySum(self, df, col):
		df = df.groupby(by = col).sum()
		df.reset_index(inplace=True)
		try:
			df.columns = ['entityName', 'lookup_freq', 'lookup_decay_frequency']
		except Exception as e:
			print(df.columns)
			sys.exit()
		lookup_dict = dict(zip(df['entityName'], df['lookup_freq']))
		lookup_decay_dict = dict(zip(df['entityName'], df['lookup_decay_frequency']))
		return lookup_dict, lookup_decay_dict
	def sigmoid(self, x):
		return 1/(1 + math.exp(-x))

class Prefetch:
	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")
		self.nlp.remove_pipe('ner')
		self.__matcher = Matcher(self.nlp.vocab, validate=True)
		self.__doc = None
		self.__programmingLanguages = None
		self.__topics = None
		self.__people = None
		self.__components = None
		self.__os = None
		self.__cloud = None
		self.__teams = None
		self.__communication_tech = None
		self.__emails_ = None
		self.file_name = sys.argv[1]
		self.user_json_file = sys.argv[2]
		self.output_df_file = sys.argv[3]
		self.functions = predefinedFunctions()


	def  create_versioned(self, name):
		return [
		[{'LOWER': name}], 
		[{'LOWER': {'REGEX': f'({name}\d+\.?\d*.?\d*)'}}], 
		[{'LOWER': name}, {'TEXT': {'REGEX': '(\d+\.?\d*.?\d*)'}}],
		]

	def __create_topic_patterns(self):
	    # versioned_languages = ['FPGA']
	    # flatten = lambda l: [item for sublist in l for item in sublist]
	    # versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	        [{'LOWER': 'pre'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'approval'}],
	        [{'LOWER': 'home'}, {'LOWER': 'value'}],
	        [{'LOWER': 'rental'}, {'LOWER': 'property'}],
	        [{'LOWER': 'home'}, {'LOWER': 'insurance'}],
	        [{'LOWER': 'mortgage'}, {'LOWER': 'rates'}],
	        [{'LOWER': 'mortgage'}, {'LOWER': 'insurance'}],
	        [{'LOWER': 'down'}, {'LOWER': 'payment'}],
	        [{'LOWER': 'down'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'payment'}],
	        [{'LOWER': 'financing'}, {'LOWER': 'rates'}],
	        [{'LOWER': 'refinancing'}, {'LOWER': 'rates'}],
	        [{'LOWER': 'finance'}, {'LOWER': 'advice'}],
	        [{'LOWER': "seller's"}, {'LOWER': 'marketplace'}],
	        [{'LOWER': "seller's"}, {'LOWER': 'market'}, {"LOWER": "place"}],
	        [{'LOWER': "senior"}, {'LOWER': 'housing'}],
	        [{'LOWER': "housing"}, {'LOWER': 'market'}, {"LOWER": "place"}],
	        [{'LOWER': "housing"}, {'LOWER': 'marketplace'}],
	        [{'LOWER': "principal"}, {'IS_PUNCT': True, 'OP':  '?'}, {"LOWER": "interest"}],
	        [{'LOWER': "home"}, {'IS_PUNCT': True, 'OP':  '?'}, {"LOWER": "owner's"}, {'IS_PUNCT': True, 'OP':  '?'}, {"LOWER": "association"}],
	        [{'LOWER': "homeowner's"}, {'IS_PUNCT': True, 'OP':  '?'}, {"LOWER": "association"}],
	        [{'LOWER': "property"}, {'LOWER': 'records'}],
	        [{'LOWER': "seller's"}, {'LOWER': 'marketplace'}],
	        [{'LOWER': "seller's"}, {'LOWER': 'marketplace'}],
	        [{'LOWER': 'broker'}],
	        [{'LOWER': 'listings'}],
	        [{'LOWER': 'listing'}],
	        [{'LOWER': 'lease'}],
	        [{'LOWER': 'tenants'}],
	        [{'LOWER': 'tenant'}],
	        [{'LOWER': 'principal'}],
	        [{'LOWER': 'interest'}],
	        [{'LOWER': 'downpayment'}],
	        [{'LOWER': 'hoa'}],
	        [{'LOWER': 'realtor'}],
	        [{'LOWER': 'foreclosure'}],
	        [{'LOWER': "fore"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'closure'}],
	        [{'LOWER': "short"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'sales'}],
	        [{'LOWER': "short"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'selling'}],
	        [{'LOWER': 'fsbo'}],
	        [{'LOWER': 'mls'}],
	        [{'LOWER': "multiple"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'listing'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'service'}],
	        [{'LOWER': "real"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'estate'}],
	        [{'LOWER': "real"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'estate'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'license'}],
	        [{'LOWER': "real"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'estate'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'listings'}],
	        [{'LOWER': "real"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'estate'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'investing'}],
	        [{'LOWER': 'trulia'}],
	        [{'LOWER': 'zillow'}],
	        [{'LOWER': 'loopnet'}],
	        [{'LOWER': 'compass'}],
	        [{'LOWER': 'movoto'}],
	        [{'LOWER': 'hotpads'}],
	        [{'LOWER': 'redfin'}],
	        [{'LOWER': "coldwell"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'banker'}],
            [{'LOWER': "sotheby's"}],
            [{'LOWER': "sotheby"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': "'s"}],
            [{'LOWER': "apartments.com"}],
            [{'LOWER': "apartments"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'com'}],
            [{'LOWER': "rent.com"}],
            [{'LOWER': "rent"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'com'}],
            [{'LOWER': "apartment"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'guide'}],
            [{'LOWER': "homes.com"}],
            [{'LOWER': "homes"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'com'}],
            [{'LOWER': "forrent.com"}],
            [{'LOWER': "forrent"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'com'}],
	        [{'LOWER': "sotheby's"}],
	        [{'LOWER': 'remax'}],
	        [{'LOWER': 'realtytrac'}],
	        [{'LOWER': "ziprealty"}],
	        [{'LOWER': "architecture"}],
	        [{'LOWER': 'flippers'}],
	        [{'LOWER': "interior"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'design'}],
	        [{'LOWER': 'interiordesign'}],
	        
	        #wordCloud
	        [{'LOWER': 'client'}],
	        [{'LOWER': 'update'}],
	        [{'LOWER': 'document'}],
	        [{'LOWER': 'service'}],
	        [{'LOWER': 'user'}],
	        [{'LOWER': 'admin'}],
	        [{'LOWER': 'inc'}],
	        [{'LOWER': 'agent'}],
	        [{'LOWER': 'update'}],
	        
	        [{'LOWER': 'id'}],
	        [{'LOWER': 'meeting'}],
	        [{'LOWER': 'business'}],
	        [{'LOWER': 'seller'}],
	        [{'LOWER': 'provide'}],
	        [{'LOWER': 'docusign'}],
	        [{'LOWER': "docu"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'sign'}],
	        [{'LOWER': 'commission'}],
	        [{'LOWER': 'cofounder'}],
	        [{'LOWER': 'founder'}],
	        
	        [{'LOWER': 'outline'}],
	        [{'LOWER': 'roadmap'}],
	        [{'LOWER': "road"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'map'}],
	        [{'LOWER': 'funding'}],
	        [{'LOWER': 'communication'}],
	        [{'LOWER': 'feature'}],
	        [{'LOWER': 'sign'}],
	        [{'LOWER': 'help'}],
	        [{'LOWER': 'property'}],
	        
	        [{'LOWER': 'reupliving'}],
	        [{'LOWER': 'organization'}],
	        [{'LOWER': "upfront"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'cost'}],
	        [{'LOWER': "upfront"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'costs'}],
	        [{'LOWER': 'renovation'}],
	        [{'LOWER': 'price'}],
	        [{'LOWER': 'address'}],
	        [{'LOWER': 'privacy'}],
	        [{'LOWER': 'support'}],
	        [{'LOWER': 'sale'}],
	        [{'LOWER': 'contact'}],
	        [{'LOWER': 'support'}],
	        [{'LOWER': 'sale'}],
	        [{'LOWER': {'IN': ['remodel', 'remodeling']}}],
	        [{'LOWER': {'IN': ['repair', 'repairs']}}],
	        [{'LOWER': {'IN': ['contractor', 'contractors']}}],
	        [{'LOWER': {'IN': ['', 'repairs']}}],
	        [{'LOWER': {'IN': ['asset', 'assets']}}],
	        [{'LOWER': {'IN': ['sell', 'selling']}}],
	        [{'LOWER': 'seller'}],
	        [{'LOWER': 'buyer'}],
	        [{'LOWER': 'market price'}],
	        [{'LOWER': 'profit'}],
	        [{'LOWER': 'budget'}],
	        [{'LOWER': 'evaluation'}],
	        [{'LOWER': 'construction'}],
	        [{'LOWER': 'home sale'}]
        
        ]
	    return topic_patterns

	def __create_os_patterns(self):
	    versioned_languages = ['windows', 'linux', 'ubuntu', 'unix', 'fedora', 'macos', 'solaris', 'centos']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	        [{'LOWER': 'windows'}],
	        [{'LOWER': 'ubuntu'}],
	        [{'LOWER': 'linux'}],
	        [{'LOWER': 'fedora'}],
	        [{'LOWER': 'macos'}],
	        [{'LOWER': 'solaris'}],
	        [{'LOWER': 'centos'}],
	        [{'LOWER': 'ios'}],
	        [{'LOWER': 'android'}],
	        [{'LOWER': 'unix'}],
	        [{'LOWER': 'mac'}, {'LOWER': 'os'}],
	    ]


	    return versioned_patterns + topic_patterns


	def __create_app_patterns(self):
	    versioned_languages = ['brosix', 'bugzilla', 'slack', 'git', 'github', 'jira', 'Sharepoint', 'gmail', 'skype', 'zoom', 'youtube', 'onedrive', 'asana', 'confluence', 'dropbox', 'flickr', 'hubspot', 'proofhub', 'basecamp', 'Teamviewer']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	        [{'LOWER': 'brosix'}],
	        [{'LOWER': 'dropbox'}], [{'LOWER': 'drop'}, {'LOWER': 'box'}],
	        [{'LOWER': 'googledrive'}], [{'LOWER': 'google'}, {'LOWER': 'drive'}],
	        [{'LOWER': 'flickr'}],
	        [{'LOWER': 'bugzilla'}],
	        [{'LOWER': 'asana'}],
	        [{'LOWER': 'slack'}],
	        [{'LOWER': 'salesforce'}],
	        [{'LOWER': 'youtube'}],
	        [{'LOWER': 'microsoftoffice'}],
	        [{'ORTH': 'microsoft365'}],
	        [{'LOWER': 'microsoft'}, {'ORTH': '365'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'office'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'teams'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'excel'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'outlook'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'word'}],
	        [{'LOWER': 'share'}, {'LOWER': 'point'}],
	        
	        [{'LOWER': 'freshdesk'}],
	        [{'LOWER': 'proofhub'}],
	        [{'LOWER': 'hubspot'}],
	        [{'TEXT': 'Notion'}],
	        [{'LOWER': 'basecamp'}],
	        [{'LOWER': 'zoom'}],
	        [{'LOWER': 'confluence'}],
	        [{'LOWER': 'invision'}],
	        [{'LOWER': 'jira'}],
	        [{'LOWER': 'gmail'}],
	        [{'LOWER': 'email'}],
	        [{'LOWER': 'outlook'}],
	        [{'LOWER': 'onedrive'}],
	        [{'LOWER': 'one'}, {'LOWER': 'drive'}],
	        [{'LOWER': 'onenote'}],
	        [{'LOWER': 'one'}, {'LOWER': 'note'}],
	        [{'LOWER': 'teamviewer'}],
	        [{'LOWER': 'team'}, {'LOWER': 'viewer'}],
	        [{'LOWER': 'git'}],
	        [{'LOWER': 'github'}],
	        [{'LOWER': 'gitlab'}],
	        [{'LOWER': 'excel'}],
	        [{'TEXT': 'Teams'}],
	        
	        
	    ]
	    return topic_patterns

	def __create_cloud_patterns(self):
	    versioned_languages = ['aws', 'gcp', 'azure']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	        [{'lower': 'aws'}],
	        [{'lower': 'azure'}],
	        [{'lower': 'gcp'}],
	        [{'lower': 'kamatera'}],
	        [{'lower': 'serverspace'}],
	        [{'lower': 'linode'}],
	        [{'lower': 'sciencesoft'}],
	        [{'lower': 'scalahosting'}],
	        [{'lower': 'cloudways'}],
	        [{'lower': 'liquidweb'}],
	        [{'lower': 'siteground'}],
	        [{'lower': 'wpengine'}],
	        [{'lower': 'digitalocean'}],
	        [{'lower': 'vultr'}],
	        [{'lower': 'navisite'}],
	        [{'LOWER': 'microsoft'}, {'LOWER': 'azure'}],
	        [{'LOWER': 'oracle'}, {'LOWER': 'cloud'}],
	        [{'LOWER': 'scala'}, {'LOWER': 'hosting'}],
	        [{'LOWER': 'wp'}, {'LOWER': 'engine'}],
	        [{'LOWER': 'verizon'}, {'LOWER': 'cloud'}],
	        [{'LOWER': 'amazon'}, {'LOWER': 'web'}, {"LOWER": 'service'}],
	        [{'LOWER': 'google'}, {'LOWER': 'cloud'}],
	        [{'LOWER': 'ibm'}, {'LOWER': 'cloud'}],
	        [{'LOWER': 'google'}, {'LOWER': 'cloud'}, {"LOWER": 'platform'}],
	        
	    ]
	    return topic_patterns

	def __create_lang_patterns(self):
	    versioned_languages = ['ruby', 'php', 'python', 'perl', 'java', 'haskell', 
	                           'scala', 'c', 'r', 'cpp', 'matlab', 'bash', 'delphi', 'jython', 'cython', 'ruby']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    lang_patterns = [
	        [{'LOWER': 'objective'}, {'IS_PUNCT': True, 'OP':  '?'},{'LOWER': 'c'}],
	        [{'LOWER': 'objectivec'}],
	        [{'LOWER': 'octave'}],
	        [{'LOWER': 'objective'}, {'IS_PUNCT': True, 'OP':  '?'},{'LOWER': 'j'}],
	        [{'LOWER': 'objectivej'}],
	        [{'LOWER': 'c'}, {'LOWER': '#'}],
	        [{'LOWER': 'c'}, {'LOWER': 'sharp'}],
	        [{'LOWER': 'c#'}],
	        [{'LOWER': 'f'}, {'LOWER': '#'}],
	        [{'LOWER': 'f'}, {'LOWER': 'sharp'}],
	        [{'LOWER': 'f#'}],
	        [{'LOWER': 'lisp'}],
	        [{'LOWER': 'common'}, {'LOWER': 'lisp'}],
	        [{'LOWER': 'go', 'POS': {'NOT_IN': ['VERB']}, 'DEP': {'NOT_IN': ['pobj']}}],
	        [{'LOWER': 'golang'}],
	        [{'LOWER': 'python'}],
	        [{'LOWER': 'php'}],
	        [{'LOWER': 'haskell'}],
	        [{'LOWER': 'delphi'}],
	        [{'LOWER': 'bash'}],
	        [{'LOWER': 'matlab'}],
	        [{'LOWER': 'cython'}],
	        [{'LOWER': 'html'}],
	        [{'LOWER': 'css'}],
	        [{'LOWER': 'sql'}],
	        [{'LOWER': 'nosql'}],
	        [{'LOWER': 'scala'}],
	        [{'LOWER': 'java'}],
	        [{'LOWER': 'jscript'}],
	        [{'LOWER': 'javafx'}],
	        [{'LOWER': 'kotlin'}],
	        [{'LOWER': 'powershell'}],
	        [{'LOWER': 'javafx'}],
	        [{'LOWER': 'swift'}],
	        [{'LOWER': {'IN': ['js', 'javascript']}}],
	        [{'LOWER': {'IN': ['ts', 'typescript', 'type script']}}],
	        [{'LOWER': 'c++'}],
	        [{'LOWER': 'r++'}],
	        [{'LOWER': 'ruby'}],
	        [{'LOWER': 'rust'}],
	        [{'LOWER': 'sas'}],
	        [{'LOWER': 'sasl'}],
	        [{'LOWER': 'spark'}],
	        [{'LOWER': 'sqr'}],
	        [{'LOWER': 'tex'}],
	        [{'LOWER': 'unixshell'}],
	        [{'LOWER': 'unix'}, {'LOWER': 'shell'}],
	        [{'LOWER': 'ubercode'}],
	        [{'LOWER': 'xquery'}],
	        [{'LOWER': 'z++'}],
	        [{'LOWER': 'zeno'}],
	        [{'LOWER': 'zebra'}],
	    ]


	    return lang_patterns


	def __create_components(self):
	    versioned_languages = ['CPX-CPU', 'DIS', 'DMA', 'DRC', 'FNN', 'GPU', 'MCE', 'MRD', 'DIA', 'HCP', 'LSP', 'MIS', 'MMU', 'NOC', 'OSB', 'PCIe', 'USB', 'TOP']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])
	    component_patterns = []
	    
	    return component_patterns


	def __create_team_patterns(self):
	    versioned_languages = ['CNN Team', 'Crypto Team Discussions', 'Communication Team', 'All hands', 'FPGA', 'Products & Marketing', 'Kirkwood Architecture', 'Gopi & Marketing', 'Corp Leaders', 'Immigration_SSR', 'Product Documentation', 'NonImmig_closedGrp', 'Tantra Analyst', 'Kirkwood Design', 'Axiado Library', 'Kirkwood FPGA', 
	                           'Design Verification', 'Axiado Exec Team', 'MAC TEAM', 'SVlabs', 'MAC and USB Team', 'AxCan', 'Kirkwood Product Lifecycle', 'sw-engineering', 'Ethernet Sub-System', 'DIA Women', 'Kirkwood SOC Build Up', 'STEM/H1B', 'Pathfinders', 'EdgeIQ-Board-and-Package-Design', 'CoreAndPeripherals','EdgeIQ Testing and Simulations', 
	                          'EdgeIQ Clock Interactions']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	    	[{'LOWER': 'scenic-oaks-trl'}],
	    	[{'LOWER': 'changing-the-real-estate-world'}],
	    	[{'LOWER': 'potential-homes'}],
	    	[{'LOWER': 'reup-marketing'}],
	    	[{'LOWER': "scenic"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'oaks'}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'trl'}],
	    	[{'LOWER': "reup"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'marketing'}],
	    	[{'LOWER': "potential"}, {'IS_PUNCT': True, 'OP':  '?'}, {'LOWER': 'homes'}],
	    ]
	    return topic_patterns

	def __create_people_patterns_full(self):
	    topic_patterns = [
			[{"LOWER": 'ryan'}, {"LOWER": "sawchuk", 'OP': '?'}],
			[{'LOWER': {"IN": ['ryan', 'sawchuk', 'ryan sawchuk']}}],

			[{"LOWER": 'andrew'}, {"LOWER": "mitchell", 'OP': '?'}],
			[{'LOWER': {"IN": ['andrew mitchell', 'andrew', 'mitchell']}}],

			[{"LOWER": 'alex'}, {"LOWER": "kassem", 'OP': '?'}],
			[{'LOWER': {"IN": ['alex kassem', 'alex', 'kassem']}}],

			[{"LOWER": 'keyan'}, {"LOWER": "chang", 'OP': '?'}],
			[{'LOWER': {"IN": ['keyan chang', 'keyan', 'chang']}}],

			[{"LOWER": 'jean'}, {"LOWER": "baptiste", 'OP': '?'}, {"LOWER": "jacquot", 'OP': '?'}],
			[{'LOWER': {"IN": ['jean baptiste jacquot', 'jean', 'baptiste', 'jb', 'jacquot']}}],

			[{"LOWER": 'erez'}, {"LOWER": "haimowicz", 'OP': '?'}],
			[{'LOWER': {"IN": ['erez haimowicz', 'erez', 'haimowicz']}}],

			[{"LOWER": 'justin'}, {"LOWER": "mattas", 'OP': '?'}],
			[{'LOWER': {"IN": ['justin mattas', 'justin', 'mattas']}}],

			#         [{"LOWER": 'jb'}, {"LOWER": "jacquot", 'OP': '?'}],
			#             [{'LOWER': {"IN": ['jb jacquot', 'jb', 'jacquot']}}],

			[{"LOWER": 'andrea'}, {"LOWER": "haimowicz", 'OP': '?'}],
			[{'LOWER': {"IN": ['andrea haimowicz', 'andrea']}}],

			[{"LOWER": 'stacy'}, {"LOWER": "summers", 'OP': '?'}],
			[{'LOWER': {"IN": ['stacy summers', 'stacy', 'summers']}}],
			[{"LOWER": 'zahid'}]
        ]


	    return topic_patterns

	# def __create_email_patterns(self):
	# 	email_flag = lambda text: bool(re.compile(r'^[^@]+@[^@]+.[^@]+$'))
	# 	IS_MY_EMAIL = self.nlp.vocab.add_flag(email_flag)
	# 	return IS_MY_EMAIL



	def inistiatePatterns(self):
		self.__programmingLanguages = self.__create_lang_patterns()
		self.__topics = self.__create_topic_patterns()
		self.__people = self.__create_people_patterns_full()
		# self.__components = self.__create_components()
		self.__os = self.__create_os_patterns()
		self.__cloud = self.__create_cloud_patterns()
		self.__teams = self.__create_team_patterns()
		self.__communication_tech = self.__create_app_patterns()
		# self.__final_topics = self.__topics + self.__components
		self.__matcher.add("PROG_SCRIPT_LANG", self.__programmingLanguages)
		self.__matcher.add("TOPICS", self.__topics)
		self.__matcher.add("PEOPLE", self.__people)
		# self.__matcher.add("COMPONENTS", self.__components)
		self.__matcher.add("OS", self.__os)
		self.__matcher.add("CLOUD_PLATFORMS", self.__cloud)
		self.__matcher.add("TEAMS", self.__teams)
		self.__matcher.add("COMMUNICATION_TECH_PLATFORMS", self.__communication_tech)

	def ent_item_extract(self, df):
	    d = dict()
	    result = {}
	    [d [t [1]].append(t [0]) if t [1] in list(d.keys()) else d.update({t [1]: [t [0]]}) for t in df]
	    for i in list(d.keys()):
	        result[i] = self.ent_item_extract2(d[i])
	    return result

	def ent_item_extract2(self, df):
	    return Counter(w for w in df)


	def people_replace(self, df):
	    people_map = {
	        "jean": "jb jacquot",
	        "jean baptiste": "jb jacquot",
	        "baptiste": "jb jacquot",
	        }
	    names2 = []
	    for i in df:

	        if i in people_map.keys():
	            names2.append(people_map[i])
	        else:
	            names2.append(i)
	    return names2
	def topic_replace(self, df):
		topic_map = {
		    'listing' : 'listings',
		    'tenants' :  'tenant',
		    'fore closure' : 'foreclosure',
		    'multiple listing service' : 'mls',
		    'interiordesign': 'interior design',
		    'reup living' : 'reupliving',
		    'remodel': 'remodeling',
		    'asset' : 'assets',
		    'repair' : 'repairs', 
		    'renovations' : 'renovation',
		    'contractors' : 'contractor',
		    "homeowner's association" : 'hoa'
		    }
		names2 = []
		for i in df:
		    
		    if i in topic_map.keys():
		        names2.append(topic_map[i])
		    else:
		        names2.append(i)
		return names2
			

	def substringReplace(self, names):
	    names2 = []
	    for i in names:
	        if len(str(i).split())==1:
	            ms = [w for w in names if w.startswith(i) or w.endswith(i)]
	        else:
	            ms = [w for w in names if w.startswith(i)]
	        result = [w for w in list(set(ms)) if len(w) == max(len(x) for x in ms)]
	        names2.append(' '.join(list(set(result))))
	    return names2


	def ent_item_extract3(self, df):
		d = dict()
		result = {}
		[d [t [1]].append(t [0]) if t [1] in list(d.keys()) else d.update({t [1]: [t [0]]}) for t in df]
		for i in list(d.keys()):
		    if i == "PEOPLE":
		    	d[i] = self.people_replace(d[i])
		    	d[i] = self.substringReplace(d[i])
		    elif i == "TOPICS":
		    	d[i] = self.topic_replace(d[i])
		    result[i] = self.ent_item_extract2(d[i])
		return result


	def remove_tail_spaces(self, df):
	    try:
	        return df.rstrip()
	    except Exception as e:
	        return ', '.join([w.rstrip() for w in df])


	def parse_entities(self, doc):
	    # doc = self.nlp(doc)
	    matches = self.__matcher(doc)
	    spans = [doc[start:end] for _, start, end in matches]
	    indexes = iter([self.nlp.vocab.strings[idx] for idx, start, end in self.__matcher(doc) if doc[start:end] in spacy.util.filter_spans(spans)])

	    detections = [(span.text, next(indexes)) for span in spacy.util.filter_spans(spans)]
	    keys = self.substringReplace([w[0] for w in detections])
	    values = [w[1] for w in detections]
	    detections = list(zip(keys, values))
	    return detections

  
	

class Files(Prefetch):

	def __init__(self):
		super().__init__()
		self.inistiatePatterns()
		self.dateAgg = None
		self.__user_interest_dict = {}
		self.__mail_to_date = {}
		with open(self.file_name, encoding="utf-8") as json_file:
			print("reading sjonl file", self.file_name)
			self.__data = json_file.read()
			self.__result = [json.loads(jline) for jline in self.__data.splitlines()]
			json_file.close()
		csv_obj = s3_client.get_object(Bucket='augmentor-customer-data', Key = 'REUP-INTERESTS/REUP_TRANSIENT.csv')
		body = csv_obj['Body']
		csv_string = body.read().decode('utf-8')
		self.__au_df_ = pd.read_csv(StringIO(csv_string))
		self.__ax = list(self.__au_df_['email (S)'])

	def vals(self, x):
	    if isinstance(x, dict):
	        for v in x.values():
	            result.extend(vals(v))
	        return result
	    else:
	        return [x]

	def jsonl_to_dict(self, jsonObj):
	    conv_df = {}
	    conv_id = []
	    mssg_id = []
	    subject = []
	    content = []
	    date = []
	    cc_recipients = []
	    to_recipients = []
	    to_id = []
	    to_name = []
	    bcc_recipients = []
	    from_email = []
	    from_name = []
	    from_id = []

	    for conv in range(len(jsonObj)):
	        for record in jsonObj[conv]:
	            conv_id.append(record['conversation_id'])
	            mssg_id.append(record['id'])
	            try:
	                subject.append(record['subject'])
	            except Exception as keyError:
	                subject.append('NA')
	            date.append(record['date'])
	            content.append(record['body'])
	            from_name.append(record['from']['user_display_name'])
	            from_id.append(record['from']['user_id'])
	            from_email.append(record['from']['email'])
	            to_recipients.append([self.vals(w['email']) for w in self.vals(record['recipients'])[0]])
	            to_id.append([self.vals(w['user_id']) for w in self.vals(record['recipients'])[0]])
	            to_name.append([self.vals(w['user_display_name']) for w in self.vals(record['recipients'])[0]])
	    conv_df['conversationId '] = conv_id
	    conv_df['fromEmailAddress '] = from_email
	    conv_df['fromName'] = from_name
	    conv_df['fromId'] = from_id
	    conv_df['toRecipients '] = to_recipients
	    conv_df['to_name'] = to_name
	    conv_df['to_id'] = to_id
	    conv_df['subject '] = subject
	    conv_df['body.content '] = content
	    conv_df['receivedDateTime '] = date
	    conv_df['id '] = mssg_id
	    return conv_df

	bad_str = r'[<>]'
	def email_clean(self, df):
	    try:
	        df_ = re.sub(bad_str, ' ', df)
	        return df_
	    except Exception as emptyBody:
	        return df

	def email_name_list(self, mail):
	    name = mail.split("@")[0]
	    name = name.replace('.', ' ')
	    name_list = ' '.join(name.split()).title()
	    return name_list

	def email_name_list_(self, mails):
		emails_list = []
		for mail in mails:
			name = mail.split("@")[0]
			name = name.replace('.', ' ')
			name_list = name.split()
			emails_list.append(name_list)
		return emails_list

	def add_IDType(self, df):
		if pd.isnull(df):
		    return 'EMAIL'
		else:
		    return 'COGNITO_SUB'
    


	def add_USER_tag(self, df):
		if pd.isnull(df):
		    return df
		elif df in self.__ax:
			return "USER_" + self.mailToId(df)
		else:
		    return 'USER_' + df  

	def data_wrangler(self, df):
		wg = []
		for i in range(len(df)):
		# df['fromEmailAddress '] = df['fromEmailAddress '].apply(lambda x: eval(x)[0])
		# df['toRecipients '] = df['toRecipients '].apply(lambda x: eval(x)[0])
			try:
				wg.append(df['subject '].loc[i] + ' ' +  df['body.content '].loc[i] + ' ' + df['toRecipients '].loc[i])
			except Exception as e:
				# print("error occured in data_wrangler")
				wg.append(df['subject '].loc[i] + ' ' +  df['body.content '].loc[i] + ' ' + ' '.join(df['toRecipients '].loc[i]))
		return wg

	def cleanhtml(self, raw_html):
		try:
		    raw_html = ' '.join([w for w in raw_html.split() if w.lower() not in stops])
		    raw_html = re.sub(r'http\S+', '', raw_html)
		    clear_punct = '!"#$%()*+-/;<=>?[\\]^_`{|}~'
		    content = raw_html.translate(str.maketrans('', '', clear_punct)).lower()
		except Exception as nullVallue:
		    pass
		return raw_html

	def __read_from_json(self, file_name):
		with open(file_name, 'r') as f:
			data = json.loads(f.read())
		f.close()
		# print("read json file: ", file_name)
		return data
	  
	def __write_to_json(self, file_name, data):
		with open(file_name, 'w') as f:
			# f.write(json.dumps(data))
			json.dump(data, f, indent = 4)
		f.close()
		# print("Written to json file: ", file_name)
	def user_interest(self, id_):
		self.__user_df = self.__interest_df[(self.__interest_df['fromEmailAddress '] == id_) | (self.__interest_df['toRecipients '].str.contains(id_))]
		user_content = list(self.__user_df['entities'])
		user_content_ = sum(user_content, [])
		try:
			self.__mail_to_date[id_] = self.__user_df['receivedDateTime '].apply(lambda x: parser.parse(x).strftime("%Y-%m-%d")).mode().loc[0]
		except Exception as keyError:
			self.__mail_to_date[id_] = None
			
		name = id_.split('@')[0]
		name = name.replace('.', ' ')
		name_list = [w.lower().rstrip() for w in name.split()]
		try:
		    bad_name1 = name.split()[0].lower() + '.'+ name.split()[1].lower()
		    bad_name2 = name.split()[0].lower()
		except Exception as e:
		    bad_name2 = name.lower()
		    bad_name1 = name.split()[0].lower()
		self.__user_content_ = [(w[0].lower().rstrip(), w[1].rstrip()) for w in user_content_ if not (w[0].lower().startswith(tuple(name_list)))]
		self.__user_counter = self.ent_item_extract3(self.__user_content_)
		self.__user_interest_dict[id_.rstrip()] = self.__user_counter

	def summer(self, l2):
		l3 = []
		vals = list(l2.values())
		l3 = [[val, "EMAIL"] for val in vals[0]]
		return l3
		
	def find_mails(self, content):
		mails = [w for w in re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z-]+\.[a-zA-Z]+)", content) if pd.isnull(w) == False and w in self.__emails_]
		return {"EMAIL" : list(set(mails))}
	def replaces(self, df):
		return df.replace(' (', '/')
	def decayFunc1(self, df, freq):
		if pd.isnull(df):
			return freq
		pre = datetime.now().strftime("%Y-%m-%d")
		pre = parser.parse(pre)
		pa = parser.parse(df)
		origin = pre - pd.DateOffset(days = 365)
		diff = pre - pa
		# print(diff.days)
		# if diff.days <=2:
		#     return math.ceil(freq * math.exp(math.log10(diff.days)*-0.01))
		# elif diff.days <= 7:
		#     return math.ceil(freq * math.exp(math.log10(diff.days)*-0.03))
		# elif diff.days <=21:
		#     return math.ceil(freq * math.exp(math.log10(diff.days)*-0.055))
		# elif diff.days <=31:
		#     return math.ceil(freq * math.exp(math.log10(diff.days)*-0.065))
		# elif diff.days <= 60:
		#     return math.ceil(freq * math.exp(math.log(diff.days)*-0.05))
		if diff.days <=7:
		    return math.ceil(freq * math.exp((math.log(7) - math.log(diff.days))*0.1025))

		elif diff.days <=21:
		    return math.ceil(freq * math.exp((math.log(21) - math.log(diff.days))*0.07))
		elif diff.days <=31:
		    return math.ceil(freq * math.exp((math.log(31) - math.log(diff.days))*0.02))

		elif diff.days <=62:
		    return math.ceil(freq * math.exp((math.log(62) - math.log(diff.days))*0.01))

		elif diff.days <=183:
		    return math.ceil(freq * math.exp(math.log(diff.days)*-0.02))
		elif diff.days <= 365:
		    return math.ceil(freq * math.exp(math.sqrt(diff.days)*-0.058))
		elif diff.days <= 365*2:
		    return math.ceil(freq * math.exp(math.sqrt(diff.days)*-0.08))
		else:
		    return math.ceil(freq * math.exp(math.sqrt(diff.days)*-0.1))

	def parser_db(self, user_interest_dict):
		sk = []
		value = []
		frequency = []
		user_id = []
		alt_value = []
		category = []
		for record in user_interest_dict:
		    for key in user_interest_dict[record]:
		        ranked_dict = dict(user_interest_dict[record][key].most_common())
		        ranked_keys = list(ranked_dict.keys())
		        ranked_rank = np.arange(len(ranked_dict))
		        for val in range(len(ranked_keys)):
		            user_id.append(record)
		            sk.append('TAG_' + key+'_'  + str(ranked_rank[val]+1).zfill(4))
		            category.append(key)
		            value.append(ranked_keys[val])
		            
		            frequency.append(ranked_dict[ranked_keys[val]])
		return user_id, sk, value, frequency, category

	def parse_decay_dict(self, user_interest_dict):
		decay_frequency = []
		for record in user_interest_dict:
		    for key in user_interest_dict[record]:
		        ranked_dict = dict(user_interest_dict[record][key].most_common())
		        ranked_keys = list(ranked_dict.keys())
		        ranked_rank = np.arange(len(ranked_dict))
		        for val in range(len(ranked_keys)):
		            decay_frequency.append(ranked_dict[ranked_keys[val]])
		return decay_frequency

	def nameToId(self, name):
	    name = name.split("@")[0]
	    name = name.replace('.', ' ')
	    name_list = name.split()
	    flag = 0
	    for i in self.__email_to_name_dict:
	        user_name_list = list(self.__email_to_name_dict[i].values())[1].lower()
	        # user_name_list = [w.lower() for w in user_name_list]
	        if len(name_list) >1:
	            if name.split()[0] in user_name_list and name.split()[1] in user_name_list:
	                flag = 1
	                break
	        else:
	            if name.split()[0] in user_name_list:
	                flag = 1
	                break
	    if flag == 1:
	        return i
	    else:
	        for i in self.__ext_email_to_name_dict:
	            enl = list(self.__ext_email_to_name_dict[i]['names'])
	            if len(name_list)>1:
	                if name.split()[0] in enl and name.split()[1] in enl:

	                    flag = 1
	                    break
	            else:
	                if name.split()[0] in enl:
	                    flag = 1
	                    break


	            
	    if flag == 1:
	        return i
	    else:
	        return name


	def nameToDisplayName(self, name):
		name = name.split("@")[0]
		name = name.replace('.', ' ')
		name_list = name.split()
		flag = 0
		for i in self.__email_to_name_dict:
		    user_name_list = list(self.__email_to_name_dict[i].values())[1].lower()
		    # user_name_list = [w.lower() for w in user_name_list]
		    if len(name_list) >1:
		        if name.split()[0] in user_name_list and name.split()[1] in user_name_list:
		            flag = 1
		            break
		    else:
		        if name.split()[0] in user_name_list:
		            flag = 1
		            break
		if flag == 1:
		    return user_name_list.title()
		else:
		    for i in self.__ext_email_to_name_dict:
		        enl = list(self.__ext_email_to_name_dict[i]['names'])
		        if len(name_list)>1:
		            if name.split()[0] in enl and name.split()[1] in enl:

		                flag = 1
		                break
		        else:
		            if name.split()[0] in enl:
		                flag = 1
		                break
		        
		if flag == 1:
		    return user_name_list.title()
		else:
		    return None

	def mailToId(self, mail):
		flag = 0
		for i in self.__email_to_name_dict:
		    user_mail_list = list(self.__email_to_name_dict[i].values())[0]
		    if len(mail) >3:
		        if mail in user_mail_list:
		            flag = 1
		            break
		if flag == 1:
		    return i
		else:
		    return mail

	def mailToName(self, mail):
		flag = 0
		for i in self.__email_to_name_dict:
		    user_name_list = list(self.__email_to_name_dict[i].values())[1].lower()
		    # user_name_list = [w.lower() for w in user_name_list]
		    user_mail_list = list(self.__email_to_name_dict[i].values())[0]
		    if len(mail) >3:
		        if mail in user_mail_list:
		            flag = 1
		            break
		            
		if flag == 1:
		    return user_name_list.title()
		else:
			if mail in self.__ext_email_to_name_dict:
				ext_name = ' '.join(self.__ext_email_to_name_dict[mail]['names'])
				return ext_name.title()
			else:
				return None

	def scaler_function(self, df, flag):
		min_ = df.min()
		max_ = df.max()
		std = (df-min_)/(max_ - min_)
		if flag == 1:
			return std
		else:
			return std * (1 - 0.1) + 0.1

	def get_param(self):
		self.__res = self.jsonl_to_dict(self.__result)
		self.__df = pd.DataFrame.from_dict(self.__res)
		# self.dateAgg = self.__df['receivedDateTime '].apply(lambda x: parser.parse(x).strftime("%Y-%m-%d")).mode().item()

		self.__df_copy = self.__df.copy()
		self.__df_copy['body.content '] = self.__df_copy['body.content '].str.lower()

		self.__df_copy['body.content '] = self.__df_copy['body.content '].apply(self.email_clean)

		self.__df_copy['toRecipients '] = self.__df_copy['toRecipients '].apply(lambda x: sum(x, []))
		self.__df_copy['to_id'] = self.__df_copy['to_id'].apply(lambda x: sum(x, []))
		self.__df_copy['to_name'] = self.__df_copy['to_name'].apply(lambda x: sum(x, []))
		self.__toAddresses_unique = list(set(sum(list(self.__df_copy['toRecipients ']), [])))

		# self.__ccAddresses_unique = list(set(sum(list(self.__df['ccRecipients ']), [])))

		self.__emails_ = list(set(list(self.__df_copy['fromEmailAddress '].unique()) + self.__toAddresses_unique))
		self.__emails_ = [w.lower() for w in self.__emails_]
		self.__wrangle_df = self.data_wrangler(self.__df_copy)
		self.__wrangle_df = pd.DataFrame(self.__wrangle_df, columns=['content'])

		self.__interest_df =  pd.concat([self.__df['fromEmailAddress '], self.__df_copy['toRecipients '], self.__df['receivedDateTime '], self.__wrangle_df], axis = 1)

		self.__interest_df['content'] = self.__interest_df['content'].apply(self.cleanhtml)

		# self.__content_pipe = self.nlp.pipe(self.__interest_df['content'])
		# self.__list_entities = []
		# for i in self.__content_pipe:
		# 	self.__list_entities.append(self.parse_entities(i))

		self.__interest_df['entities'] = [self.parse_entities(w) for w in self.nlp.pipe(self.__interest_df['content'])]		

		# self.__interest_df['entities'] = self.__interest_df['content'].apply(self.parse_entities)
		# self.__interest_df['entities'] = self.__list_entities

		self.__interest_df['mails_ext'] = self.__interest_df['content'].apply(self.find_mails)
		self.__interest_df['mails_ext'] = self.__interest_df['mails_ext'].apply(self.summer)

		self.__interest_df['entities'] = self.__interest_df['entities'] + self.__interest_df['mails_ext']

		self.__interest_df['fromEmailAddress '] = self.__interest_df['fromEmailAddress '].apply(self.remove_tail_spaces)
		self.__interest_df['toRecipients '] = self.__interest_df['toRecipients '].apply(self.remove_tail_spaces)
		self.__interest_df['fromEmailAddress '] = self.__interest_df['fromEmailAddress '].str.lower()
		self.__interest_df['toRecipients '] = self.__interest_df['toRecipients '].str.lower()

		# self.__interest_df['ccRecipients '] = self.__interest_df['ccRecipients '].apply(self.remove_tail_spaces)
		self.__output_df = self.__interest_df.to_dict('records')

		# self.__write_to_json("obj_interset_df.json", self.__output_df)

		self.__lookup_emails = [w for w in self.__emails_ if w.endswith('@reupliving.com') or w in self.__ax]
# 		print("lookup_mails: ", self.__lookup_emails)
		for i in self.__lookup_emails:
		    if len(i)>3:
		        self.user_interest(i)

		self.__user_interest_dict_copy = copy.deepcopy(self.__user_interest_dict)

		for i in self.__user_interest_dict_copy:
			for j in self.__user_interest_dict_copy[i]:
				for k in self.__user_interest_dict_copy[i][j]:
					decay_freq = self.__user_interest_dict_copy[i][j][k]
					self.__user_interest_dict_copy[i][j][k] = self.decayFunc1(self.__mail_to_date[i], decay_freq)


		# self.__natest = self.__user_interest_dict
		# self.__write_to_json("obj_user_interest.json", self.__natest)
		# flag = 0
		try:
			print("Opening existing obj_user_interest.json")
			obj_ = s3_client.get_object(Bucket='augmentor-customer-data', Key='REUP-INTERESTS/obj_user_interest.json')
			data = obj_['Body'].read().decode('utf-8')
			if len(data) == 0:
			    print("File has no content and program is terminated")
			    sys.exit()

			self.__uid = json.loads(data)

			print("Opening existing obj_user_interest_decay.json")
			obj_ = s3_client.get_object(Bucket='augmentor-customer-data', Key='REUP-INTERESTS/obj_user_interest_decay.json')
			data = obj_['Body'].read().decode('utf-8')
			if len(data) == 0:
			    print("File has no content and program is terminated")
			    sys.exit()

			self.__decay_uid = json.loads(data)

			for i in self.__uid:
				for j in self.__uid[i]:
					self.__uid[i][j] = Counter(self.__uid[i][j])

			for i in self.__decay_uid:
				for j in self.__decay_uid[i]:
					self.__decay_uid[i][j] = Counter(self.__decay_uid[i][j])

			# print("Changed the dict to Counter dict")
			self.__scrap = []
			for i in self.__user_interest_dict:
			    if i in self.__uid:
			        self.__scrap.append(i)
			print()
			# print("Found interesect in dict.keys()")
			print('===============TRY===============')
			# print(self.__user_interest_dict['amit.patel@axiado.com'])
			# print('==================================')
			for i in self.__scrap:
			    for j in self.__uid[i]:
			        if j == "PEOPLE" and j in self.__user_interest_dict[i]:
			            puid = self.substringReplace(list(self.__uid[i][j].keys()) + list(self.__user_interest_dict[i][j].keys()))
			            tuid = self.__uid[i][j]
			            vuid = self.__user_interest_dict[i][j]
			            self.__uid[i][j] = Counter(dict(zip(puid[:len(tuid)], list(tuid.values()))))
			            self.__user_interest_dict[i][j] = Counter(dict(zip(puid[len(tuid):], list(vuid.values()))))
			            self.__uid[i][j] = self.__user_interest_dict[i][j] + self.__uid[i][j]
			        elif j!= 'PEOPLE' and j in self.__user_interest_dict[i]:
			            self.__uid[i][j] = self.__user_interest_dict[i][j] + self.__uid[i][j]

			for i in self.__scrap:
			    for j in self.__decay_uid[i]:
			        if j == "PEOPLE" and j in self.__user_interest_dict_copy[i]:
			            puid = self.substringReplace(list(self.__decay_uid[i][j].keys()) + list(self.__user_interest_dict_copy[i][j].keys()))
			            tuid = self.__decay_uid[i][j]
			            vuid = self.__user_interest_dict_copy[i][j]
			            self.__decay_uid[i][j] = Counter(dict(zip(puid[:len(tuid)], list(tuid.values()))))
			            self.__user_interest_dict_copy[i][j] = Counter(dict(zip(puid[len(tuid):], list(vuid.values()))))
			            self.__decay_uid[i][j] = self.__user_interest_dict_copy[i][j] + self.__decay_uid[i][j]
			        elif j!= 'PEOPLE' and j in self.__user_interest_dict_copy[i]:
			            self.__decay_uid[i][j] = self.__user_interest_dict_copy[i][j] + self.__decay_uid[i][j]

			self.__natest = dict(list(self.__user_interest_dict.items()) + list(self.__uid.items()))
			self.__decay_natest = dict(list(self.__user_interest_dict_copy.items()) + list(self.__decay_uid.items()))

			s3_client.put_object(Body=json.dumps(self.__natest), Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/obj_user_interest.json')
			s3_client.put_object(Body=json.dumps(self.__decay_natest), Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/obj_user_interest_decay.json')

		except Exception as e:
			print('==========EXCEPT===========')
			# print(self.__user_interest_dict['amit.patel@axiado.com'])
			# print('==================================')
			self.__decay_natest = self.__user_interest_dict_copy

			self.__natest = self.__user_interest_dict
			s3_client.put_object(Body=json.dumps(self.__natest), Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/obj_user_interest.json')
			s3_client.put_object(Body=json.dumps(self.__decay_natest), Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/obj_user_interest_decay.json')


		user_id, sk, value, frequency, category = self.parser_db(self.__natest)
		decay_frequency = self.parse_decay_dict(self.__decay_natest)

		final_df = pd.DataFrame(user_id, columns=['email_id'])
		final_df['sk'] = sk
		final_df['value'] = value
		final_df['category'] = category
		final_df['frequency'] = frequency
		final_df['decay_frequency'] = decay_frequency

		final_df['sk'] = final_df['sk'].apply(self.replaces)
		final_df['value'] = final_df['value'].apply(self.replaces)
		fna = final_df.copy()

		# au_df_ = pd.read_csv('AUGMENTOR_TRANSIENT2.csv')
		self.__au_df_ = self.__au_df_[['cognitoSubjectId (S)', 'email (S)', 'displayName (S)']]
		self.__au_df_.columns = ['sub_id', 'email_id', 'userName']
		self.__au_df_['userName'] = self.__au_df_['userName']

		self.__emails_list = self.email_name_list_(self.__emails_)

		self.__email_to_name_dict = {}
		for i in range(len(self.__au_df_)):
		    self.__email_to_name_dict[self.__au_df_['sub_id'].loc[i]] = {"emailId" : self.__au_df_['email_id'].loc[i], "names" : self.__au_df_['userName'].loc[i]}
		    


		self.__ext_email_to_name_dict = {}
		for i in range(len(self.__emails_)):
		    self.__ext_email_to_name_dict[self.__emails_[i]] = {"names" : self.__emails_list[i]}


		self.__sova_people = fna[fna['category'] == "PEOPLE"]['value'].apply(self.nameToId)
		fna.loc[fna.category == "PEOPLE", 'alt_value'] = self.__sova_people

		# self.__full_tr = self.__read_from_json("final_fake_transient.json")

		self.__sova_avatar = fna[fna['category'] == "PEOPLE"]['value'].apply(self.nameToDisplayName)
		fna.loc[fna.category == "PEOPLE", 'alt_avatar'] = self.__sova_avatar


		# self.__sova_avatar = fna[fna['category'] == "PEOPLE"]['value'].apply(self.nameToDisplayName)
		fna.loc[fna.category == "PEOPLE", 'entityLabel'] = "Mention"

		self.__sova_email = fna[fna['category'] == "EMAIL"]['value'].apply(self.mailToId)
		fna.loc[fna.category == "EMAIL", 'alt_value'] = self.__sova_email

		self.__sova_avatar = fna[fna['category'] == "EMAIL"]['value'].apply(self.mailToName)
		fna.loc[fna.category == "EMAIL", 'alt_avatar'] = self.__sova_avatar

		fna.loc[fna.category == "EMAIL", 'entityLabel'] = "Mail"

		final_au_int_df = final_df.merge(self.__au_df_, on='email_id', how='left')
		final_au_int_df['id_type'] = final_au_int_df['sub_id'].apply(self.add_IDType)
		# final_au_int_df['userName'].fillna(final_au_int_df['email_id'].apply(self.email_name_list), inplace=True)
		final_au_int_df['sub_id'].fillna(final_au_int_df['email_id'], inplace=True)
		final_au_int_df['sub_id'] = final_au_int_df['sub_id'].apply(self.add_USER_tag)
		final_au_int_df['nameToId'] = fna['alt_value']
		final_au_int_df['entityName'] = fna['alt_avatar']
		final_au_int_df['entityName'].fillna(final_au_int_df['value'].apply(lambda x: x.title()), inplace = True)
		final_au_int_df['decay_frequency'] = final_df['decay_frequency']
		final_au_int_df['entityLabel'] = fna['entityLabel']
		final_au_int_df['category'] = final_au_int_df['category'].replace('COMPONENTS', 'TOPICS')
		final_au_int_df['category'] = final_au_int_df['category'].replace('EMAIL', 'PEOPLE')
		final_au_int_df['nameToId'] = final_au_int_df['nameToId'].apply(self.add_USER_tag)

		df_cumulative = final_au_int_df.copy()
		print(df_cumulative.columns)
		df_cumulative['userName'] = df_cumulative['userName'].apply(lambda x: x.title() if pd.isnull(x)==False else None)
		df_cumulative = df_cumulative[(df_cumulative['category'] == 'TOPICS') | (df_cumulative['category'] == 'PEOPLE') | (df_cumulative['category'] == 'TEAMS') | (df_cumulative['category'] == 'PROG_SCRIPT_LANG')]
		frequency_lookup, decay_lookup = self.functions.groupbySum(df_cumulative, 'entityName')
		orphan_lookup = self.functions.groupbySum(df_cumulative, 'nameToId')
		df_cumulative['self_freq'] = df_cumulative[df_cumulative['category'] == 'PEOPLE']['userName'].apply(lambda x: frequency_lookup[x] if pd.isnull(x)==False else None)
		df_cumulative['self_decay_freq'] = df_cumulative[df_cumulative['category'] == 'PEOPLE']['userName'].apply(lambda x: decay_lookup[x] if pd.isnull(x)==False else None)
		df_cumulative['freq_lookup'] = df_cumulative['entityName'].apply(lambda x: frequency_lookup[x])
		# df_cumulative['decay_lookup'] = df_cumulative['entityName'].apply(lambda x: decay_lookup[x])
		df_cumulative['lookup_freq'] = df_cumulative['entityName'].apply(lambda x: frequency_lookup[x])
		df_cumulative['lookup_decay_freq'] = df_cumulative['entityName'].apply(lambda x: decay_lookup[x])
		df_cumulative['self_freq'].fillna(df_cumulative['frequency'], inplace = True)
		df_cumulative['self_decay_freq'].fillna(df_cumulative['decay_frequency'], inplace = True)
		df_cumulative['total_freq'] = df_cumulative['lookup_freq'] + df_cumulative['self_freq'].fillna(0)
		df_cumulative['total_decay_freq'] = df_cumulative['lookup_decay_freq'] + df_cumulative['self_decay_freq'].fillna(0)
		df_cumulative['total_freq'] = df_cumulative['total_freq'].apply(lambda x: math.sqrt(x))
		df_cumulative['self_freq'] = df_cumulative['frequency'].apply(lambda x: math.log(x))
		df_cumulative['total_decay_freq'] = df_cumulative['total_decay_freq'].apply(lambda x: math.sqrt(x))
		df_cumulative['self_decay_freq'] = df_cumulative['decay_frequency'].apply(lambda x: math.log(x))
		# df_cumulative['total_freq'] = df_cumulative['self_freq'] + df_cumulative['self_freq'].fillna(0)
		# df_cumulative[['self_normalized', 'total_normalized']] = pfs.normalize(df_cumulative[['frequency', 'total_freq']])
		df_ = df_cumulative.copy()
		# df_ = df_[(df_['category'] == 'TOPICS') | (df_['category'] == 'PEOPLE') | (df_['category'] == 'TEAMS') | (df_['category'] == 'PROG_SCRIPT_LANG')]
		# df_[['sc_frequency', 't_frequency']] = self.functions.normalize(df_[['self_freq', 'total_freq']])
		df_['sc_frequency'] = self.scaler_function(df_['self_freq'], 0)
		df_['t_frequency'] = self.scaler_function(df_['total_freq'], 0)
		df_['scd_frequency'] = self.scaler_function(df_['self_decay_freq'], 0)
		df_['td_frequency'] = self.scaler_function(df_['total_decay_freq'], 0)
		df_['weight'] = 1.2* df_['sc_frequency'] + df_['t_frequency']
		df_['d_weight'] = 1.2* df_['scd_frequency'] + df_['td_frequency']
		# df_['weight'] = self.functions.scaler.fit_transform(df_[['weight']])
		df_['weight'] = self.scaler_function(df_[['weight']], 1)
		df_['weight'] = df_['weight'].apply(lambda x: round(x, 3))
		df_['d_weight'] = self.scaler_function(df_[['d_weight']], 1)
		df_['d_weight'] = df_['d_weight'].apply(lambda x: round(x, 3))

		df_.drop(columns = ['sc_frequency', 't_frequency', 'td_frequency', 'scd_frequency', 'freq_lookup', 'self_freq', 'self_decay_freq', 'total_freq', 'total_decay_freq'], inplace = True)
		test_interest_json = df_.to_json(orient = 'records', lines = True)
		s3_client.put_object(Body=test_interest_json, Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/testInterest.jsonl')

		res_df = final_au_int_df.to_json(orient = 'records', lines = True)

		resultLength = len(df_)
		print("length: ", resultLength)
		df_slice_first = 0
		df_slice_second = 2
		cnt = 0
		if resultLength < 2000:
			outputInterestsKey = 'REUP-INTERESTS/outDir/testInterest_0.jsonl' 
			res_df_ = df_[df_slice_first*1000:df_slice_second*1000].to_json(orient = 'records', lines = True)
			s3_client.put_object(Body= res_df_, Bucket = 'augmentor-customer-data', Key = outputInterestsKey)
		else:
		    for outputInteresIndex, outputInterestsData in enumerate(df_[df_slice_first*1000:df_slice_second*1000]):
		        if resultLength/1000 < df_slice_first:
		            break
		            
		        else:
		        	outputInterestsKey = 'REUP-INTERESTS/outDir/testInterest_' + outputInteresIndex + '.jsonl'
		        	res_df_ = df_[df_slice_first*1000:df_slice_second*1000].to_json(orient = 'records', lines = True)
		        	s3_client.put_object(Body= res_df_, Bucket = 'augmentor-customer-data', Key = outputInterestsKey)
		        	df_slice_first = df_slice_second
		        	df_slice_second = df_slice_second+2
		        	time.sleep(0.5)

		csv_buf = StringIO()
		final_au_int_df.to_csv(csv_buf, header=True, index=False)
		csv_buf.seek(0)
		s3_client.put_object(Bucket='augmentor-customer-data', Body=csv_buf.getvalue(), Key='REUP-INTERESTS/final_au_int_df_update.csv')

		s3_client.put_object(Body= res_df, Bucket = 'augmentor-customer-data', Key = 'REUP-INTERESTS/final_au_int_df_update.jsonl')
		print("written to final_au_int_df_update.json and csv")
		print('Completed writing to csv file')

	def displayParams(self):
		print(self.__interest_df.head())

			
if __name__ == "__main__":
	obj = Prefetch()
	obj.inistiatePatterns()
	gfile = Files()
	gfile.get_param()
