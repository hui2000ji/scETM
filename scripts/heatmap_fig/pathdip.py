# Note: Works with Python 3 and up

import urllib.request, urllib.parse


# ########################################################################################################
#                                                   class library mirDIP_Http                            #
# ########################################################################################################
class pathDIP_Http:


    url = "http://ophid.utoronto.ca/pathDIP/Http_API"
        
    map = {}    # results will be here

    def __init__(self):
        return


    def searchOnUniprot_IDs(self, IDs, component, sources):
        self.sendPost('Uniprot ID', IDs, component, sources)
        return

    def searchOnGenesymbols(self, IDs, component, sources):
        self.sendPost('Gene Symbol', IDs, component, sources)
        return

    def searchOnEntrez_IDs(self, IDs, component, sources):
        self.sendPost('Egid', IDs, component, sources)
        return


    # .. serve POST request
    def sendPost(self, typeChoice, IDs, component, sources):

        params = {
        'typeChoice' : typeChoice,
        'IDs' : IDs,
        'TableName' : component,
        'DataSet' : sources}

        params = bytes( urllib.parse.urlencode( params ).encode() )
        response = ''

        try:
            handler = urllib.request.urlopen(self.url, params)
        except Exception:
            traceback.print_exc()
        else:
            self.response = handler.read().decode('utf-8')
            ## print(self.response)
            self.makeMap()

        return


    def makeMap(self):
            
        ENTRY_DEL = 0x01
        KEY_DEL = 0x02
            
        arr = self.response.split(chr(ENTRY_DEL))
            
        for str in arr:
                
            arrKeyValue = str.split(chr(KEY_DEL));
            if len(arrKeyValue) > 1: 
                self.map[arrKeyValue[0]] = arrKeyValue[1]

        return


    def getGeneratedAt(self): 

        if "GeneratedAt" in self.map: 
            return self.map["GeneratedAt"]
        else:
            return ''

    def getIDs(self):
       
        if "IDs" in self.map:
            return self.map["IDs"]
        else:
            return ''

    def getDataComponent(self): 
        if "TableName" in self.map: 
            return self.map["TableName"]
        else:
            return ''

    def getSources(self): 
        if "DataSet" in self.map: 
            return self.map["DataSet"]
        else:
            return ''

    def getPathwayAnalysisSize(self):
        if "SummarySize" in self.map: 
            return self.map["SummarySize"]
        else: 
            return ''

    def getPathwayAnalysis(self):
        if "Summary" in self.map: 
            return self.map["Summary"]
        else:
            return '' 

    def getDetailsSize(self):
        if "DetailsSize" in self.map: 
            return self.map["DetailsSize"]
        else: 
            return ''

    def getDetails(self): 
        if "Details" in self.map: 
            return self.map["Details"]
        else: 
            return ''





#########################################################################################################
#                                          Pick-up right sample for your search                         #
#########################################################################################################
# Note: Adjust 'ophid.utoronto.ca' if you are using our development server.

        

# ###########################################
#      Example of search on Uniprot IDs     #
# ###########################################

# Uniprot IDs 
#      - Comma delimited. 
#      - Mind case.
IDs = "O15379,P15692,P13236,P13236,P13236,Q96P48,O00468"
        
# Data component  
#      - Use the only one of those five:
#          Literature curated (core) pathway memberships
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.95
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.95
#      - Mind exact spelling and spaces.
component = "Literature curated (core) pathway memberships"
        
# Data sources  (Note: Data source list was updated for full list of 22 available sources)  
#      - Use some or all of those:
#          ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways
#      - Comma delimited.
#      - Mind exact spelling.
sources = "ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways"
            
            
            
o = pathDIP_Http()
o.searchOnUniprot_IDs(IDs, component, sources)

# print results
print("\r\n  Search on Uniprot IDs:  \r\n")

print("Generated at: " + o.getGeneratedAt())
print("IDs: " + o.getIDs())
print("DataComponent: " + o.getDataComponent())
print("Sources: " + o.getSources())

print();
print("Pathway enrichment analysis results size: " + o.getPathwayAnalysisSize());
print("Pathway enrichment analysis results ('q-value (FDR: BH-method) less than 0.05'): \r\n" + o.getPathwayAnalysis());  # formatted as tab-delimited spreadsheet

print("Detailed table size: " + o.getDetailsSize());
print("Pathway annotations for the full input list: \r\n" + o.getDetails());  # formatted as tab-delimited spreadsheet




        

# ###########################################
#      Example of search on Gene Symbols    #
# ###########################################

# Gene Symbols 
#      - Use HUGO Gene nomenclature
#      - Comma delimited. 
#      - Mind case.
IDs = "HDAC3, VEGFA, CCL4, DP141L, ARAP1, MARK4, ZZZ3, AGRN, MAPRKE1, CRABP1, HIST1H1C, RACGAPI"
        
# Data component  
#      - Use the only one of those five:
#          Literature curated (core) pathway memberships
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.95
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.95
#      - Mind exact spelling and spaces.
component = "Literature curated (core) pathway memberships"
        
# Data sources  (Note: Data source list was updated for full list of 22 available sources)  
#      - Use some or all of those:
#          ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways
#      - Comma delimited.
#      - Mind exact spelling.
sources = "ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways"
            
            
            
o = pathDIP_Http()
o.searchOnGenesymbols(IDs, component, sources);

# print results
print("\r\n  Search on Gene Symbols:  \r\n")

print("Generated at: " + o.getGeneratedAt())
print("IDs: " + o.getIDs())
print("DataComponent: " + o.getDataComponent())
print("Sources: " + o.getSources())

print();
print("Pathway enrichment analysis results size: " + o.getPathwayAnalysisSize())
print("Pathway enrichment analysis results ('q-value (FDR: BH-method) less than 0.05'): \r\n" + o.getPathwayAnalysis())  # formatted as tab-delimited spreadsheet

print("Detailed table size: " + o.getDetailsSize())
print("Pathway annotations for the full input list: \r\n" + o.getDetails())  # formatted as tab-delimited spreadsheet




        

# ###########################################
#    Example of search on Entrez Gene IDs   #
# ###########################################

# Entrez Gene ID 
#      - Comma delimited. 
#      - Mind case.)
IDs = "375790,8841,6351,7422"
        
# Data component  
#      - Use the only one of those five:
#          Literature curated (core) pathway memberships
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.95
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.99
#          Extended pathway associations. Protein interaction set: Experimentally detected and computationally predicted PPIs (full IID)Minimum confidence level for predicted associations: 0.95
#      - Mind exact spelling and spaces.
component = "Extended pathway associations. Protein interaction set: Experimentally detected PPIsMinimum confidence level for predicted associations: 0.99"
        
# Data sources    # Data sources  (Note: Data source list was updated for full list of 22 available sources)  
#      - Use some or all of those:
#          ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways
#      - Comma delimited.
#      - Mind exact spelling.
sources = "ACSN2,BioCarta,EHMN,HumanCyc,INOH,IPAVS,KEGG,NetPath,OntoCancro,Panther_Pathway,PharmGKB,PID,RB-Pathways,REACTOME,stke,systems-biology.org,SignaLink2.0,SIGNOR2.0,SMPDB,Spike,UniProt_Pathways,WikiPathways"
            
            
o = pathDIP_Http()
o.searchOnEntrez_IDs(IDs, component, sources)

# print results
print("\r\n  Search on Entrez Gene ID:  \r\n")

print("Generated at: " + o.getGeneratedAt())
print("IDs: " + o.getIDs())
print("DataComponent: " + o.getDataComponent())
print("Sources: " + o.getSources())

print();
print("Pathway enrichment analysis results size: " + o.getPathwayAnalysisSize())
print("Pathway enrichment analysis results ('q-value (FDR: BH-method) less than 0.05'): \r\n" + o.getPathwayAnalysis())  # formatted as tab-delimited spreadsheet

print("Details Size: " + o.getDetailsSize())
print("Pathway annotations for the full input list: \r\n" + o.getDetails())  # formatted as tab-delimited spreadsheet