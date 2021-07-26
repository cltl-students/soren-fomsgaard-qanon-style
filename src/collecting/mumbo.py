### THIS SCRIPT IS AN ADAPTATION OF CODE PROVIDED BY TOM DE SMEDT.
#%%
import re
import os

#from grasp import *

#%% My import modifications
# this cell is only needed when running in ipython.

# import sys

#sys.path.append(os.path.abspath('../../src')) # append top level dir to sys path.

from utils.grasp import * # import grasp modules.
#from utils.collect import queries



#%%
#data = csv(cd('some/dir/file.csv')) # modified to point to Data dir.
#seen = set(col(0, data))

# Queries below are modified to fit the nonQanon google sheet.
# q = (
#     'from:WHO',
#     'from:MSF',
#     'from:genocide_watch',
#     'from:wikileaks',
#     'from:NatGeo',
#     'from:tim_cook',
#     'from:Wiccateachings',
#     'from:NASA',
#     'from:Lenore_Black_',
#     'from:OGOMProject',
#     'from:JoeUscinski',
#     'from:QOrigins',
#     'from:willsommer',
#     'from:travis_view',
#     'from:RightWingWatch',
#     'from:jaredholt',
#     'from:DFRLab',
#     'from:AustSkeptics',
#     'from:blackskeptics',
#     'from:CDCgov',
#     'from:coe',
#     'from:fact_covid',
#     'from:factcheckdotorg',
#     'from:AapFactcheck',
#     'from:psyop_debunker',
#     'from:qnonbeliever',
#     'from:BBCWorld',
#     'from:StopFakingNews',
#     'from:ACSHorg',
#     'from:factsimilie',
#     'from:DebunkTrump',
#     'from:MuslimCouncil',
#     'from:JCUA_News',
#     'from:NCJW',
#     'from:HolocaustMuseum',
#     'from:HolocaustCtr',
#     'from:QAnon_Decrypted',
#     'from:jsrailton',
#     'from:RichardDawkins',
#     'from:FairytalesFood',
#     'from:LTNorseMyths',
# )

#%%

def run_queries(q, file):
    """Run Twitter username queires against Twitter API.

    Args:
        q (tuple): A tuple of query strings.
        file (str): A str filepath to a save results.
    """  
    data = csv(cd(file)) # modified to point to Data dir.
    seen = set(col(0, data))
 
    for q in reversed(q):
        for t in twitter(q):
            if t.id not in seen:
                data.append((
                    t.id,
                    t.author,
                    t.language,
                    t.text,
                    t.date,
                    t.likes,
                ))
                seen.add(t.id)

        data.save()
#TODO: add check / counter or timer.

#%%

#UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
#
#for r in shuffled(data):
#    for f in re.findall(r'https://pbs.twimg.com/[^\s]+\.jpg', r[3]):
#        k = cd('mumbo', f.split('/')[-1])
#        if not os.path.exists(k):
#            try:
#                v = download(f, delay=0, headers={'User-Agent': UA})
#                f = open(k, 'wb')
#                f.write(v)
#                f.close()
#                print(k)
#            except Exception as e:
#                print(e)

# %%
