import spacy
import pandas as pd
import numpy as np
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
import json
import os
import io
import pickle
from collections import Counter
import re
import sys
import time

stops = ['i', 're:', 'fwd:', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


class Prefetch:
	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")
		self.nlp.remove_pipe('ner')
		self.__matcher = Matcher(self.nlp.vocab, validate=True)
		self.__doc = self.nlp("My name is Vincent and I was born on 23rd June 1987. \
		   I work at Rasa from Haarlem. I just bought a guitar \
		   cost $1000 on ebay and I will get is services here for 20 euro a year.")
		self.__programmingLanguages = None
		self.__topics = None
		self.__people = None
		self.__components = None
		self.__os = None
		self.__cloud = None
		self.__teams = None
		self.__communication_tech = None
		self.__emails_ = None


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
	        [{'LOWER': 'edge'}, {"LOWER": "iq", 'op': '+'}],
	        [{'LOWER': {"IN": ['edgeiq', 'edge iq']}}],
	        # [{'LOWER': 'edgeiq'}],
	        [{'LOWER': 'watchguard'}],
	        [{'LOWER': 'sdk'}],
	        
	        # [{'LOWER': 'cyberattack'}],
	        [{'LOWER': 'cyber'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'attack'}],
	        [{'LOWER': {"IN": ['cyber attack', 'cyberattack']}}],
	        [{'LOWER': 'iiot'}],
	        
	        [{'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'ai'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "engine"}],
	        [{'LOWER': 'sve'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'vault'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "extension"}],
	        
	        [{'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'vault'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "extension"}],
	        [{'LOWER': 'sae'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'ai'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "engine"}],
	        
	        # [{"LOWER": "rocketchip"}],
	        [{'LOWER': 'rocket'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "chip"}],
	        [{'LOWER': {"IN": ['rocket chip', 'rocketchip']}}],
	        [{'LOWER': 'cloud'}],
	        
	        [{"LOWER": "soc"}],
	        [{"LOWER": 'system'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "on"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'chip'}],
	        [{'LOWER': 'soc'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'system'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "on"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'chip'}],
	        [{"LOWER": "rtl"}],
	        
	        [{'LOWER': 'zap'}],
	        [{'LOWER': 'full'}, {"IS_PUNCT":True}, {"LOWER": "stack"}, {"IS_PUNCT":True}, {"LOWER": "zap"}],
	        [{'LOWER': 'fullstack'}, {"LOWER": "zap"}],
	        
	        [{"LOWER": "asic"}],
	        [{"LOWER": "asic"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'application'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'specific'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "integrated"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "circuit"}],
	        [{'LOWER': 'application'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'specific'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "integrated"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "circuit"}],
	        [{"LOWER": "silicon"}],
	        [{"LOWER": "gds"}],
	        [{"LOWER": 'graphic'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "data"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'stream'}],
	        [{'LOWER': 'gds'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'graphic'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "data"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'stream'}],
	        [{"LOWER": "asic"}],
	        [{"LOWER": 'register'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "transfer"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'level'}],
	        [{'LOWER': 'rtl'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'register'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "transfer"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'level'}],
	        
	        [{"LOWER": "securevault"}],
	        [{'LOWER': 'svt'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "securevault"}],
	        
	        [{'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'vault'}],
	        [{'LOWER': 'svt'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'secure'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'vault'}],
	        
	        
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
	    component_patterns = [
	        [{'LOWER': 'cpx'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'cpu'}],
	        [{'LOWER': 'cpx'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'cpu'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'complex'}],
	        [{'LOWER': 'cpx-cpu'}],
	        [{'LOWER': 'dma'}],
	        [{'LOWER': {'IN': ['mis-miscellaneous', 'mis miscellaneous']}}],
	        [{'LOWER': 'drc'}],
	        [{'LOWER': 'fnn'}],
	        [{'LOWER': 'gpu'}],
	        [{'LOWER': 'hcp'}],
	        [{'LOWER': 'lsp'}],
	        [{'LOWER': 'mis'}],
	        [{'LOWER': 'mmu'}],
	        [{'LOWER': 'mce'}],
	        [{'LOWER': 'noc'}],
	        [{'LOWER': 'osb'}],
	        [{'LOWER': 'mrd'}],
	        [{'LOWER': 'pcie'}],
	        [{'LOWER': 'usb'}],
	        [{'LOWER': 'inference'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'engine'}],
	        [{'LOWER': 'firewall'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'engine'}],
	        [{'LOWER': 'memory'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'controller'}],
	        [{'LOWER': 'dia'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'planning'}],
	        
	        [{'LOWER': 'display'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'controller'}],
	        [{'LOWER': 'dis'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'display'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'controller'}],
	    
	        [{"LOWER": 'memory'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "crypto"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'engine'}],
	        [{'LOWER': 'mce', 'OP': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'memory'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "crypto"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'engine'}],
	        
	        [{"LOWER": 'direct'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "memory"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'access'}],
	        [{'LOWER': 'dma'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'direct'}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": "memory"}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'access'}],
	        [{'LOWER': 'dram'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'controller'}],
	        [{'LOWER': 'drc'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'dram'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'controller'}],
	        [{'LOWER': 'graphics'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'processing'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "unit"}],
	        [{'LOWER': 'gpu'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'graphics'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'processing'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "unit"}],
	        [{'LOWER': 'low'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'speed'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "peripheral"}],
	        [{'LOWER': 'lsp'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'low'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'speed'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "peripheral"}],
	        [{'LOWER': 'neural'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'network'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}],
	        [{'LOWER': 'fnn'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'neural'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'network'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}],
	        [{'LOWER': 'memory'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'management'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "unit"}],
	        [{'LOWER': 'mmu'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'memory'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'management'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "unit"}],
	        
	        [{'LOWER': 'network'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'on'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "chip"}],
	        [{'LOWER': 'noc'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'network'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'on'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "chip"}],
	        
	        [{'LOWER': 'on'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'chip'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "staging"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "buffer"}],
	        [{'LOWER': 'osb'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'on'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'chip'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "staging"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "buffer"}],
	        
	        [{'LOWER': 'header'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "crypto"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}],
	        [{'LOWER': 'hcp'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'header'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "crypto"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}],
	        
	        [{'LOWER': 'firewall', 'OP': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "neural"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "network"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}],
	        [{'LOWER': 'fnn', 'OP': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'firewall', 'OP': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "neural"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "network"}, {"IS_PUNCT": True, "OP": '?'}, {"LOWER": "processor"}]]
	    
	    return component_patterns


	def __create_team_patterns(self):
	    versioned_languages = ['CNN Team', 'Crypto Team Discussions', 'Communication Team', 'All hands', 'FPGA', 'Products & Marketing', 'Kirkwood Architecture', 'Gopi & Marketing', 'Corp Leaders', 'Immigration_SSR', 'Product Documentation', 'NonImmig_closedGrp', 'Tantra Analyst', 'Kirkwood Design', 'Axiado Library', 'Kirkwood FPGA', 
	                           'Design Verification', 'Axiado Exec Team', 'MAC TEAM', 'SVlabs', 'MAC and USB Team', 'AxCan', 'Kirkwood Product Lifecycle', 'sw-engineering', 'Ethernet Sub-System', 'DIA Women', 'Kirkwood SOC Build Up', 'STEM/H1B', 'Pathfinders', 'EdgeIQ-Board-and-Package-Design', 'CoreAndPeripherals','EdgeIQ Testing and Simulations', 
	                          'EdgeIQ Clock Interactions']
	    flatten = lambda l: [item for sublist in l for item in sublist]
	    versioned_patterns = flatten([self.create_versioned(lang) for lang in versioned_languages])

	    topic_patterns = [
	        [{'LOWER': 'cnn'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team'}],
	        [{'LOWER': 'crypto'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team'}],
	        [{'LOWER': 'communication'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'discussion', 'OP': '?'}],
	        [{'LOWER': 'axiado'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'exec'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team', 'OP': '?'}],
	        [{'LOWER': 'mac'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team'}],
	        [{'LOWER': 'mac'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'usb'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team', 'OP': '?'}],
	        [{'LOWER': 'kirkwood'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'product'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'lifecycle'}],
	        [{'LOWER': 'ethernet'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'sub'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'system'}],
	        [{'LOWER': 'ethernet'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'subsystem'}],

	        [{'LOWER': 'kirkwood'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'soc'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'build'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'up', 'OP': '?'}],
	        [{'LOWER': 'dia'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'women'}],
	        [{'LOWER': 'sw'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'engineering'}],
	        [{'LOWER': 'all'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'hands'}],
	        [{'LOWER': 'fpga'}],
	        [{'LOWER': 'axcan'}],
	        [{'LOWER': 'coreandperipherals'}],
	        [{'LOWER': 'core'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'peripherals'}],
	        [{'LOWER': 'dia'}],
	        [{'LOWER': 'pathfinders'}],
	        [{'LOWER': 'edgeiq clock interactions'}],
	        [{'LOWER': 'edgeiq testing and simulations'}],
	        [{'LOWER': 'edgeiq-board-and-package-design'}],
	        [{'LOWER': 'edgeiq'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'board', 'op': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', "OP": '?'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'package', 'OP': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'design', 'OP': '+'} ],
	        [{'LOWER': 'edgeiq'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'testing', 'op': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'and', "OP": '?'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'simulations', 'op': '+'}],
	        [{'LOWER': 'edgeiq'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'clock', 'op': '+'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'interactions', 'op': '+'}],
	        [{'LOWER': 'svlabs'}],
	        [{'TEXT': 'stem/h1b'}],
	        [{'LOWER': 'sv'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'labs'}],
	        [{'LOWER': 'stem'}, {"IS_PUNCT": True, 'OP': '?'}, {'TEXT': 'h1b'}],
	        [{'LOWER': 'path'}, {"IS_PUNCT": True, 'OP': '?'}, {'TEXT': 'finders'}],
	        
	        [{'TEXT': 'gopi & marketing'}],
	        [{'LOWER': 'corp'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'leaders', 'op': '+'}],
	        [{'LOWER': 'kirkwood'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'architecture', 'op': '+'}],
	        [{'LOWER': 'immigration'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'ssr', 'op': '+'}],
	        [{'LOWER': 'product'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'documentation', 'op': '+'}],
	        [{'LOWER': 'nonimmig'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'closedgrp', 'op': '+'}],
	        [{'LOWER': 'tantra'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'analyst', 'op': '+'}],
	        [{'LOWER': 'kirkwood'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'design', 'op': '+'}],
	        [{'LOWER': 'kirkwood'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'fpga', 'op': '+'}],
	        [{'LOWER': 'communication'}, {"IS_PUNCT": True, 'OP': '?'}, {'LOWER': 'team', 'op': '+'}],
	        [{'TEXT': 'axiado'}, {"IS_PUNCT": True, 'OP': '?'}, {'TEXT': 'Library', 'op': '+'}],
        

	        
	    ]
	    return topic_patterns

	def __create_people_patterns_full(self):
	    topic_patterns = [
	            [{"LOWER": 'akshat'}, {"LOWER": "karnwal", 'OP': '?'}],
	            [{'LOWER': {"IN": ['akshat karnwal', 'karnwal']}}],
	            [{"LOWER": 'vamshi'}, {"LOWER": "mahendrakar", 'OP': '?'}],
	            [{'LOWER': {"IN": ['vamshi mahendrakar', 'vamshi', 'mahendrakar']}}],
	        
	            [{"LOWER": 'sean'}, {"LOWER": "campeau", 'OP': '?'}],
	            [{'LOWER': {"IN": ['sean  campeau', 'sean', 'campeau']}}],
	            
	            [{"LOWER": 'john'}, {"LOWER": "carr", 'OP': '?'}],
	            [{'LOWER': {"IN": ['john carr', 'carr', 'john']}}],
	        
	            [{"LOWER": 'shteryana'}],
	        
	            [{"LOWER": 'kristof'}, {"LOWER": "provost", 'OP': '?'}],
	            [{'LOWER': {"IN": ['kristof provost', 'kristof', 'provost']}}],
	        
	            [{"LOWER": 'rishelle'}, {"LOWER": "zertuche", 'OP': '?'}],
	            [{'LOWER': {"IN": ['rishelle zertuche', 'rishelle', 'zertuche']}}],
	        
	            [{"LOWER": 'parnitha'}, {"LOWER": "badre", 'OP': '?'}],
	            [{'LOWER': {"IN": ['parinita badre', 'parnitha', 'badre']}}],
	        
	            [{"LOWER": 'neharika'}, {"LOWER": "yadav", 'OP': '?'}],
	            [{'LOWER': {"IN": ['neharika', 'yadav']}}],
	        
	            [{"LOWER": 'sayali'}, {"LOWER": "naik", 'OP': '?'}],
	            [{'LOWER': {"IN": ['sayali naik', 'sayali']}}],
	        
	            [{"LOWER": 'nick'}, {"LOWER": "obrien", 'OP': '?'}],
	            [{'LOWER': {"IN": ['nick obrien', 'nick', 'obrien']}}],
	        
	            [{"LOWER": 'gibran'}, {"LOWER": "khan", 'OP': '?'}],
	            [{'LOWER': {"IN": ['gibran', 'khan']}}],
	        
	            [{"LOWER": 'prasanna'}, {"LOWER": "sane", 'OP': '?'}],
	            [{'LOWER': {"IN": ['prasanna sane', 'prasanna', 'sane']}}],
	        
	            [{"LOWER": 'naga'}, {"LOWER": "maddala", 'OP': '?'}],
	            [{'LOWER': {"IN": ['naga maddala', 'naga', 'maddala']}}],
	        
	            [{"LOWER": 'tom'}, {"LOWER": "farmer", 'OP': '?'}],
	            [{'LOWER': {"IN": ['tom farmer', 'tom', 'farmer']}}],
	        
	            [{"LOWER": 'akhhiila'}, {"LOWER": "gurram", 'OP': '?'}],
	            [{'LOWER': {"IN": ['akhhiila gurram', 'akhhiila', 'gurram']}}],
	        
	            [{"LOWER": 'syam'}, {"LOWER": "sunder", 'OP': '?'}],
	            [{'LOWER': {"IN": ['syam sunder', 'syam', 'sunder']}}],
	        
	            [{"LOWER": 'dianne'}, {"LOWER": "hilly", 'OP': '?'}],
	            [{'LOWER': {"IN": ['dianne hilly', 'dianne', 'hilly']}}],
	        
	            [{"LOWER": 'divya'}, {"LOWER": "ravindran", 'OP': '?'}],
	            [{'LOWER': {"IN": ['divya ravindran', 'divya', 'ravindran']}}],
	        
	            [{"LOWER": 'alfredo'}, {"LOWER": "herrera", 'OP': '?'}],
	            [{'LOWER': {"IN": ['alfredo herrera', 'alfredo', 'herrera']}}],
	        
	            [{"LOWER": 'madhavendra'}, {"LOWER": "bhatnagar", 'OP': '?'}],
	            [{'LOWER': {"IN": ['madhavendra bhatnagar', 'madhavendra', 'bhatnagar']}}],
	        
	            [{"LOWER": 'esteban'}, {"LOWER": "figueroa", 'OP': '?'}],
	            [{'LOWER': {"IN": ['esteban figueroa', 'figueroa', 'esteban']}}],
	            [{"LOWER": 'monaco user'}],
	            [{'LOWER': 'shteryana'}],

	            [{"LOWER": 'vaidehi'},  {"LOWER": "np", 'OP': '?'}],
	            [{'LOWER': 'vaidehi'}],

	            [{'LOWER': {"IN": ['dubai user']}}],
	            [{'LOWER': 'pranav'}],
	            [{'LOWER': "vaidehi"}, {"IS_PUNCT": True, 'OP': '?'}, {"LOWER": 'np', 'OP': '?'}],
	            [{'LOWER': {"IN": ['ramiel silva', 'ramiel']}}],
	            [{"LOWER": 'ramiel'},  {"LOWER": "silva", 'OP': '?'}],
	            [{"LOWER": 'rishul'},  {"LOWER": "naik", 'OP': '?'}],
	            [{"LOWER": 'amit'}, {"LOWER": "patel", 'OP': '?'}],
	            [{'LOWER': 'psangam'}],


	            [{"LOWER": 'deepansh'}, {"LOWER": "agrawal", 'OP': '?'}],
	            [{'LOWER': {"IN": ['deepansh agrawal', 'agrawal', 'deepansh']}}],

	            [{"LOWER": 'phil'}, {"LOWER": "bujold", 'OP': '?'}],
	            [{'LOWER': {"IN": ['phil', 'bujold']}}],


	            [{'LOWER': {"IN": ['olympia bakalis', 'bakalis']}}],

	            [{"LOWER": 'prasanna'}, {"LOWER": "reddy", 'OP': '?'}],
	            [{'LOWER': {"IN": ['prasanna reddy']}}],

	            [{"LOWER": 'shreyas'}, {"LOWER": "shah", 'OP': '?'}],
	            [{'LOWER': {"IN": ['shreyas shah']}}],

	            [{"LOWER": 'shiddlingappa'}, {"LOWER": "gadad", 'OP': '?'}],
	            [{'LOWER': {"IN": ['shiddlingappa gadad', 'gadad']}}],

	            [{'LOWER': 'fahad'}],

	            [{"LOWER": 'tirumala'},{"LOWER": "marri", 'OP': '?'}],
	            [{"LOWER": 'sindhuja'},  {"LOWER": "yadiki", 'OP': '?'}],
	            [{'LOWER': {"IN": ['sindhuja yadiki', 'yadiki']}}],

	            [{"LOWER": 'aditya'},  {"LOWER": "venkatraman", 'OP': '?'}],
	            [{'LOWER': {"IN": ['aditya venkatraman', 'venkatraman']}}],
	            [{'LOWER': 'ksrao'}],
	            [{'LOWER': 'mrudhvika'}],

	            [{"LOWER": 'monaco'},  {"LOWER": "conference", 'OP': '?'}],
	            [{'LOWER': {"IN": ['monaco conference room']}}],

	            [{"LOWER": 'mike'}, {"LOWER": "lieberenz", 'OP': '?'}],
	            [{'LOWER': {"IN": ['mike lieberenz', 'lieberenz']}}],

	            [{"LOWER": 'mahesh'},  {"LOWER": "mogilisetty", 'OP': '?'}],
	            [{'LOWER': {"IN": ['mahesh mogilisetty', 'mogilisetty']}}],

	            [{"LOWER": 'gopi', 'OP': '+'},  {"LOWER": "sirineni", 'OP': '?'}],
	            [{'LOWER': {"IN": ['gopi sirineni', 'sirineni']}}],

	            [{"LOWER": 'sam'},  {"LOWER": "sandbote", 'OP': '?'}],
	            [{'LOWER': {"IN": ['sam sandbote', 'sandbote']}}],

	            [{"LOWER": 'philip'},  {"LOWER": "kearney", 'OP': '?'}],
	            [{'LOWER': {"IN": ['philip kearney', 'kearney']}}],

	            [{"LOWER": 'tejas'},  {"LOWER": "karelia", 'OP': '?'}],
	            [{'LOWER': {"IN": ['tejas karelia', 'karelia']}}],

	            [{"LOWER": 'yash'},  {"LOWER": "patel", 'OP': '?'}],
	            [{"LOWER": 'kumar'}, {"LOWER": "bhattaram", 'OP': '?'}],
	            [{'LOWER': {"IN": ['kumar bhattaram', 'bhattaram']}}],

	            [{"LOWER": 'chris'}, {"LOWER": "scott", 'OP': '?'}],
	            [{'LOWER': {"IN": ['chris scott', 'scott']}}],

	            [{"LOWER": 'patrick'},  {"LOWER": "okeeffe", 'OP': '?'}],
	            [{'LOWER': {"IN": ['patrick okeeffe', 'okeeffe']}}],


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
		self.__components = self.__create_components()
		self.__os = self.__create_os_patterns()
		self.__cloud = self.__create_cloud_patterns()
		self.__teams = self.__create_team_patterns()
		self.__communication_tech = self.__create_app_patterns()
		self.__final_topics = self.__topics + self.__components
		self.__matcher.add("PROG_SCRIPT_LANG", self.__programmingLanguages)
		self.__matcher.add("TOPICS", self.__final_topics)
		self.__matcher.add("PEOPLE", self.__people)
		# self.__matcher.add("COMPONENTS", self.__components)
		self.__matcher.add("OS", self.__os)
		self.__matcher.add("CLOUD_PLATFORMS", self.__cloud)
		# IS_MY_EMAIL = self.__create_email_patterns()
		# self.__matcher.add('EMAIL', [[{IS_MY_EMAIL: True}]])
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
	            d[i] = self.substringReplace(d[i])
	        result[i] = self.ent_item_extract2(d[i])
	    return result


	def remove_tail_spaces(self, df):
	    try:
	        return df.rstrip()
	    except Exception as e:
	        return ', '.join([w.rstrip() for w in df])


	def parse_entities(self, doc):
	    doc = self.nlp(doc)
	    matches = self.__matcher(doc)
	    spans = [doc[start:end] for _, start, end in matches]
	    indexes = iter([self.nlp.vocab.strings[idx] for idx, start, end in self.__matcher(doc) if doc[start:end] in spacy.util.filter_spans(spans)])

	    detections = [(span.text, next(indexes)) for span in spacy.util.filter_spans(spans)]
	    keys = self.substringReplace([w[0] for w in detections])
	    values = [w[1] for w in detections]
	    detections = list(zip(keys, values))
	    return detections

  
	

class Files(Prefetch):

	def __init__(self, file_name):
		super().__init__()
		self.inistiatePatterns()
		self.__user_interest_dict = {}
		with open(file_name, encoding="utf-8") as json_file:
			print("reading sjonl file", file_name)
			self.__data = json_file.read()
			self.__result = [json.loads(jline) for jline in self.__data.splitlines()]
			json_file.close()
		self.__au_df_ = pd.read_csv("AUGMENTOR_TRANSIENT2.csv")
		self.__ax = list(self.__au_df_['email (S)'])

	def jsonl_to_dict(self, jsonObj):
	    conv_df = {}
	    conv_id = []
	    mssg_id = []
	    subject = []
	    content = []
	    date = []
	    cc_recipients = []
	    to_recipients = []
	    bcc_recipients = []
	    from_email = []
	    from_name = []

	    for conv in jsonObj:
	        for record in jsonObj[conv]:
	            conv_id.append(record['conversation_id'])
	            mssg_id.append(record['id'])
	            try:
	                subject.append(record['subject'])
	            except Exception as keyError:
	                subject.append('NA')
	            date.append(record['date'])
	            content.append(record['body'])
	            from_name.append(record['from_name'])
	            from_email.append(record['from_email'])
	            to_recipients.append(record['to_recipients'])
	            cc_recipients.append(record['cc_recipients'])
	    conv_df['conversationId '] = conv_id
	    conv_df['fromEmailAddress '] = from_email
	    conv_df['toRecipients '] = to_recipients
	    conv_df['subject '] = subject
	    conv_df['body.content '] = content
	    conv_df['receivedDateTime '] = date
	    conv_df['ccRecipients '] = cc_recipients
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
	    name_list = name.split()
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
		else:
		    return 'USER_' + df  

	def data_wrangler(self, df):
		wg = []
		for i in range(len(df)):
		# df['fromEmailAddress '] = df['fromEmailAddress '].apply(lambda x: eval(x)[0])
		# df['toRecipients '] = df['toRecipients '].apply(lambda x: eval(x)[0])
			try:
				wg.append(df['subject '].loc[i] + ' ' +  df['body.content '].loc[i] + ' ' + df['toRecipients '].loc[i] + ' ' + df['ccRecipients '].loc[i])
			except Exception as e:
				# print("error occured in data_wrangler")
				wg.append(df['subject '].loc[i] + ' ' +  df['body.content '].loc[i] + ' ' + ' '.join(df['toRecipients '].loc[i]) + ' ' + ' '.join(df['ccRecipients '].loc[i]))
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
	    self.__user_df = self.__interest_df[(self.__interest_df['fromEmailAddress '] == id_) | (self.__interest_df['toRecipients '].str.contains(id_)) | (self.__interest_df['ccRecipients '].str.contains(id_))]
	    user_content = list(self.__user_df['entities'])
	    user_content_ = sum(user_content, [])
	    
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

	def nameToId(self, name):
	    name = name.split("@")[0]
	    name = name.replace('.', ' ')
	    name_list = name.split()
	    flag = 0
	    for i in self.__email_to_name_dict:
	        user_name_list = list(self.__email_to_name_dict[i].values())[1]
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
		    user_name_list = list(self.__email_to_name_dict[i].values())[1]
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
		    user_name_list = list(self.__email_to_name_dict[i].values())[1]
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
    

	def get_param(self, file_id):
		self.__res = self.jsonl_to_dict(self.__result[0])
		self.__df = pd.DataFrame.from_dict(self.__res)


		self.__df_copy = self.__df.copy()
		self.__df_copy['body.content '] = self.__df_copy['body.content '].str.lower()

		self.__df_copy['body.content '] = self.__df_copy['body.content '].apply(self.email_clean)

		# print(self.__df_copy.columns)
		# print(self.__df_copy.head())


		self.__toAddresses_unique = list(set(sum(list(self.__df['toRecipients ']), [])))

		self.__ccAddresses_unique = list(set(sum(list(self.__df['ccRecipients ']), [])))

		self.__emails_ = list(set(list(self.__df_copy['fromEmailAddress '].unique()) + self.__toAddresses_unique + self.__ccAddresses_unique))
		self.__wrangle_df = self.data_wrangler(self.__df)
		self.__wrangle_df = pd.DataFrame(self.__wrangle_df, columns=['content'])

		self.__interest_df =  pd.concat([self.__df['fromEmailAddress '], self.__df['toRecipients '], self.__df['ccRecipients '], self.__wrangle_df], axis = 1)

		self.__interest_df['content'] = self.__interest_df['content'].apply(self.cleanhtml)

		self.__interest_df['entities'] = self.__interest_df['content'].apply(self.parse_entities)

		self.__interest_df['mails_ext'] = self.__interest_df['content'].apply(self.find_mails)
		self.__interest_df['mails_ext'] = self.__interest_df['mails_ext'].apply(self.summer)

		self.__interest_df['entities'] = self.__interest_df['entities'] + self.__interest_df['mails_ext']

		self.__interest_df['fromEmailAddress '] = self.__interest_df['fromEmailAddress '].apply(self.remove_tail_spaces)
		self.__interest_df['toRecipients '] = self.__interest_df['toRecipients '].apply(self.remove_tail_spaces)
		self.__interest_df['ccRecipients '] = self.__interest_df['ccRecipients '].apply(self.remove_tail_spaces)
		self.__output_df = self.__interest_df.to_dict('records')

		# self.__write_to_json("obj_interset_df.json", self.__output_df)

		self.__lookup_emails = [w for w in self.__emails_ if w.endswith('@axiado.com') or w in self.__ax]

		for i in self.__lookup_emails:
		    if len(i)>3:
		        self.user_interest(i)

		# self.__natest = self.__user_interest_dict
		# self.__write_to_json("obj_user_interest.json", self.__natest)
		# flag = 0
		try:
			self.__uid = self.__read_from_json("obj_user_interest.json")
			print("Loaded uid.json")
			flag = 1
			# except Exception as file_error:
			# 	print("Couldn't read file")
			# 	if file_id == 0:
			# 		self.__natest = self.__user_interest_dict
			# 		self.__write_to_json("obj_user_interest.json", self.__natest)
			# 	else:
			# 		return

			# if flag == 1:

				# print(self.__uid['amit.patel@axiado.com'])
			for i in self.__uid:
				for j in self.__uid[i]:
					self.__uid[i][j] = Counter(self.__uid[i][j])

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
			self.__natest = dict(list(self.__user_interest_dict.items()) + list(self.__uid.items()))
			self.__write_to_json("obj_user_interest.json", self.__natest)
			self.__write_to_json("backup_obj_user_interest.json", self.__natest)


		except Exception as e:
			print('==========EXCEPT===========')
			# print(self.__user_interest_dict['amit.patel@axiado.com'])
			# print('==================================')

			self.__natest = self.__user_interest_dict
			self.__write_to_json("obj_user_interest.json", self.__natest)


		user_id, sk, value, frequency, category = self.parser_db(self.__natest)

		final_df = pd.DataFrame(user_id, columns=['email_id'])
		final_df['sk'] = sk
		final_df['value'] = value
		final_df['category'] = category
		final_df['frequency'] = frequency

		final_df['sk'] = final_df['sk'].apply(self.replaces)
		final_df['value'] = final_df['value'].apply(self.replaces)
		fna = final_df.copy()

		# au_df_ = pd.read_csv('AUGMENTOR_TRANSIENT2.csv')
		self.__au_df_ = self.__au_df_[['cognitoSubjectId (S)', 'email (S)', 'displayName (S)']]
		self.__au_df_.columns = ['sub_id', 'email_id', 'name']
		self.__au_df_['name'] = self.__au_df_['name'].str.lower()

		self.__emails_list = self.email_name_list_(self.__emails_)

		self.__email_to_name_dict = {}
		for i in range(len(self.__au_df_)):
		    self.__email_to_name_dict[self.__au_df_['sub_id'].loc[i]] = {"emailId" : self.__au_df_['email_id'].loc[i], "names" : self.__au_df_['name'].loc[i]}
		    


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
		final_au_int_df['sub_id'].fillna(final_au_int_df['email_id'], inplace=True)
		final_au_int_df['sub_id'] = final_au_int_df['sub_id'].apply(self.add_USER_tag)
		final_au_int_df['nameToId'] = fna['alt_value']
		final_au_int_df['entityName'] = fna['alt_avatar']
		final_au_int_df['entityName'].fillna(final_au_int_df['value'].apply(lambda x: x.title()), inplace = True)
		final_au_int_df['entityLabel'] = fna['entityLabel']
		final_au_int_df['category'] = final_au_int_df['category'].replace('COMPONENTS', 'TOPICS')
		final_au_int_df['category'] = final_au_int_df['category'].replace('EMAIL', 'PEOPLE')
		final_au_int_df['nameToId'] = final_au_int_df['nameToId'].apply(self.add_USER_tag)

		res_df = final_au_int_df.to_dict('records')
		final_au_int_df.to_csv('obj_fake_eng_df.csv', index=False)
		final_au_int_df.to_json('obj_fake_eng_df.json', orient='records')
		print('Completed writing to csv file')

	def displayParams(self):
		print(self.__interest_df.head())

			
if __name__ == "__main__":
	obj = Prefetch()
	obj.inistiatePatterns()
	doc = "I had a meeting with gopi sirineni from axiado edgeiq who is working on neural-network processor and secure vault and he wanted us to focus on cyber attack and also fix some bugs on sdk.  manojpraveen445@gmail.com That would be salome work edge iq input"
	print(obj.parse_entities(doc))

	filenames = []

	for file_name in [file for file in os.listdir() if file.endswith('.jsonl')]:
		filenames.append(file_name)


	for i in range(len(filenames)):
		gfile = Files(filenames[i])
		gfile.get_param(i)
	# gfile = Files('conversations-gson-04.14.13.56.44.jsonl')
	# gfile.get_param(0)
	# gfile.displayParams()
