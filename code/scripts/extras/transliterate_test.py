import sys
sys.path.append("..")

from indictrans import Transliterator

trn_eng2hin = Transliterator(source='eng', target='hin')
eng = ["Thats a sentence!! Aapke shubh chintakk lalit jaiswal ke taraf se aapko aapki jeet ki hardik subhkamnaye",
       "bahut",
       "baaahut",
       "bhut",
       "aman ssnti kayam kare",
       "aunty'on",
       "@DrKumarVishwas Badi Behen Haar Gai ab debate ka kya hoga.!!! ;) @thekiranbedi ab Tamasha nai ho "
       "payega...lol #AAPSweep #AAPKiDilli",
       "h, hei hey hai k p"
       ]
hin = [trn_eng2hin.transform(e) for e in eng]
print(hin)

print("")

trn_hin2eng = Transliterator(source='hin', target='eng')
hin = ["3- salman खान salman खान salman खान bollywood के biggest superstar हैं .",
       "आप जितना is tool को use करेंगे उतना ही-@@#@@ये!!आपको पसंद आयेगा .",
       "विदxxxिशा -#से बी--जेपी की सु!षमा स्वराज आगे",
       "Thats a sentence!! Aapke-shubh chintak!!@",
       "@DrKumarVishwas Badi Behen Haar Gai ab debate ka kya hoga.!!! ;) @thekiranbedi ab Tamasha nai ho "
       "payega...lol #AAPSweep #AAPKiDilli"]
eng = [trn_hin2eng.transform(h) for h in hin]
print(eng)
