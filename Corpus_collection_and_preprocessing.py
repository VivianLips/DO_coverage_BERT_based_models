!pip install requests beautifulsoup4

import requests
import time
import os
from bs4 import BeautifulSoup
import spacy
from transformers import AutoTokenizer
import json
import re


PMC_IDS = [
    'PMC10114484',
    'PMC10189931',
    'PMC10221139',
    'PMC10252790',
    'PMC10265150',
    'PMC10311057',
    'PMC10312822',
    'PMC10424294',
    'PMC10452283',
    'PMC10464436',
    'PMC10491316',
    'PMC10508771',
    'PMC10570562',
    'PMC10575590',
    'PMC10588739',
    'PMC10645424',
    'PMC10646374',
    'PMC10705299',
    'PMC10711447',
    'PMC10717560',
    'PMC10742133',
    'PMC10742326',
    'PMC10769298',
    'PMC10771463',
    'PMC10779104',
    'PMC10846626',
    'PMC10872502',
    'PMC1088418',
    'PMC10887744',
    'PMC10888680',
    'PMC10920178',
    'PMC10937278',
    'PMC10941657',
    'PMC10969498',
    'PMC10999406',
    'PMC10999556',
    'PMC11002008',
    'PMC11040453',
    'PMC11046574',
    'PMC11070171',
    'PMC11115693',
    'PMC11132695',
    'PMC11147146',
    'PMC11186997',
    'PMC11203980',
    'PMC11228553',
    'PMC11232095',
    'PMC11258377',
    'PMC11263352',
    'PMC11264201',
    'PMC11302561',
    'PMC11353077',
    'PMC11354652',
    'PMC11496539',
    'PMC11505302',
    'PMC11507888',
    'PMC11509077',
    'PMC11544622',
    'PMC11554651',
    'PMC11594678',
    'PMC11619133',
    'PMC11638135',
    'PMC11638589',
    'PMC11645674',
    'PMC11663782',
    'PMC11677501',
    'PMC11679604',
    'PMC11683280',
    'PMC11685121',
    'PMC11724961',
    'PMC11734337',
    'PMC11743456',
    'PMC11771657',
    'PMC11774304',
    'PMC1180558',
    'PMC11809137',
    'PMC1181626',
    'PMC11843878',
    'PMC11853302',
    'PMC11853562',
    'PMC11885306',
    'PMC11885495',
    'PMC11961123',
    'PMC11989206',
    'PMC12022661',
    'PMC12030754',
    'PMC1238743',
    'PMC127674',
    'PMC130486',
    'PMC1384418',
    'PMC138934',
    'PMC139260',
    'PMC1397819',
    'PMC145348',
    'PMC1456383',
    'PMC150971',
    'PMC151619',
    'PMC15863',
    'PMC174162',
    'PMC1762815',
    'PMC1779594',
    'PMC1867679',
    'PMC1876905',
    'PMC1885563',
    'PMC1892579',
    'PMC1903519',
    'PMC1907221',
    'PMC1927082',
    'PMC1948102',
    'PMC198555',
    'PMC1989115',
    'PMC199587',
    'PMC2032481',
    'PMC2032500',
    'PMC2032539',
    'PMC2042511',
    'PMC209392',
    'PMC2095280',
    'PMC2134983',
    'PMC2137364',
    'PMC2194797',
    'PMC2223455',
    'PMC2259102',
    'PMC2291517',
    'PMC2396613',
    'PMC2408405',
    'PMC2430143',
    'PMC2438280',
    'PMC2465528',
    'PMC2562179',
    'PMC2592530',
    'PMC2597945',
    'PMC2606902',
    'PMC2631358',
    'PMC2634020',
    'PMC2646245',
    'PMC2647981',
    'PMC2652633',
    'PMC2661150',
    'PMC2682445',
    'PMC2693875',
    'PMC2713530',
    'PMC2715883',
    'PMC2735333',
    'PMC2743519',
    'PMC2749008',
    'PMC2767200',
    'PMC2787016',
    'PMC2816673',
    'PMC2816827',
    'PMC2819737',
    'PMC2828043',
    'PMC2845087',
    'PMC2851946',
    'PMC2858486',
    'PMC285848665',
    'PMC2869850',
    'PMC2886993',
    'PMC2908770',
    'PMC2912286',
    'PMC2918084',
    'PMC2923685',
    'PMC2935664',
    'PMC2937732',
    'PMC2947206',
    'PMC2952981',
    'PMC2964326',
    'PMC2966797',
    'PMC2974900',
    'PMC2976316',
    'PMC29856',
    'PMC2987858',
    'PMC2993650',
    'PMC3000459',
    'PMC3014107',
    'PMC3018661',
    'PMC3057167',
    'PMC3060506',
    'PMC3080867',
    'PMC3087714',
    'PMC309134',
    'PMC3098813',
    'PMC3131109',
    'PMC3151293',
    'PMC3159892',
    'PMC3172242',
    'PMC3176311',
    'PMC3220723',
    'PMC3245036',
    'PMC3245289',
    'PMC3258534',
    'PMC3266183',
    'PMC3276674',
    'PMC3291637',
    'PMC3308137',
    'PMC3309075',
    'PMC3314481',
    'PMC3316645',
    'PMC3319163',
    'PMC33221',
    'PMC3332147',
    'PMC3340484',
    'PMC3350147',
    'PMC3366902',
    'PMC3377273',
    'PMC3378983',
    'PMC3384412',
    'PMC3386812',
    'PMC3388149',
    'PMC3392036',
    'PMC3400933',
    'PMC3425387',
    'PMC3439043',
    'PMC3440682',
    'PMC34613',
    'PMC3463903',
    'PMC3468399',
    'PMC3475759',
    'PMC3480659',
    'PMC3486727',
    'PMC3495715',
    'PMC3499097',
    'PMC3532152',
    'PMC3534255',
    'PMC3555804',
    'PMC3582397',
    'PMC3594684',
    'PMC3597531',
    'PMC3603496',
    'PMC3617971',
    'PMC3620092',
    'PMC3654184',
    'PMC3656718',
    'PMC3659335',
    'PMC3668132',
    'PMC3706435',
    'PMC3711362',
    'PMC3715917',
    'PMC3716047',
    'PMC3721468',
    'PMC3727645',
    'PMC3734370',
    'PMC3738444',
    'PMC3741462',
    'PMC3744014',
    'PMC3756378',
    'PMC3759880',
    'PMC3761935',
    'PMC3769647',
    'PMC3771870',
    'PMC3785403',
    'PMC3801070',
    'PMC3814886',
    'PMC3820303',
    'PMC3833258',
    'PMC3838615',
    'PMC3843236',
    'PMC3848588',
    'PMC3848617',
    'PMC3858585',
    'PMC3879311',
    'PMC3880612',
    'PMC3917239',
    'PMC3932673',
    'PMC3932758',
    'PMC3945669',
    'PMC3971121',
    'PMC3988845',
    'PMC4004585',
    'PMC4009955',
    'PMC403698',
    'PMC4073977',
    'PMC4076992',
    'PMC4104173',
    'PMC4142963',
    'PMC4163554',
    'PMC4169248',
    'PMC4194029',
    'PMC4220011',
    'PMC4222185',
    'PMC4227205',
    'PMC4233403',
    'PMC4234487',
    'PMC4256923',
    'PMC4273123',
    'PMC4275687',
    'PMC4286170',
    'PMC4287166',
    'PMC4304257',
    'PMC4306719',
    'PMC4319916',
    'PMC4337461',
    'PMC4348697',
    'PMC4357780',
    'PMC4359747',
    'PMC4361040',
    'PMC4363328',
    'PMC4382251',
    'PMC4395138',
    'PMC4412965',
    'PMC4422213',
    'PMC4425051',
    'PMC4426292',
    'PMC4431313',
    'PMC4442130',
    'PMC4444535',
    'PMC4447314',
    'PMC4515258',
    'PMC4537371',
    'PMC4556372',
    'PMC4557159',
    'PMC4558956',
    'PMC4572001',
    'PMC4574016',
    'PMC4576736',
    'PMC4581614',
    'PMC4585255',
    'PMC4599213',
    'PMC4602048',
    'PMC4614707',
    'PMC4629007',
    'PMC4631794',
    'PMC4632165',
    'PMC4700483',
    'PMC4702337',
    'PMC4706679',
    'PMC4725003',
    'PMC4727450',
    'PMC4743078',
    'PMC4765695',
    'PMC4776415',
    'PMC4783992',
    'PMC4787904',
    'PMC4794227',
    'PMC4802485',
    'PMC4863310',
    'PMC4864458',
    'PMC4870087',
    'PMC4875200',
    'PMC4875775',
    'PMC4906405',
    'PMC4918140',
    'PMC496690',
    'PMC4966937',
    'PMC5001238',
    'PMC5040774',
    'PMC5047687',
    'PMC5057391',
    'PMC5061611',
    'PMC5061762',
    'PMC5066219',
    'PMC5070631',
    'PMC5071152',
    'PMC507438',
    'PMC5074924',
    'PMC5085771',
    'PMC5087824',
    'PMC5095923',
    'PMC5109986',
    'PMC5121262',
    'PMC5159745',
    'PMC524294',
    'PMC5297585',
    'PMC5298918',
    'PMC5320567',
    'PMC5339079',
    'PMC5368691',
    'PMC5371890',
    'PMC5391913',
    'PMC5399479',
    'PMC5405197',
    'PMC5413206',
    'PMC5425660',
    'PMC5448071',
    'PMC5473253',
    'PMC5504018',
    'PMC5520285',
    'PMC5529005',
    'PMC5555984',
    'PMC5568783',
    'PMC5601672',
    'PMC5643014',
    'PMC5643370',
    'PMC5659706',
    'PMC5686184',
    'PMC5690827',
    'PMC5705060',
    'PMC5725889',
    'PMC5753220',
    'PMC5753380',
    'PMC5760319',
    'PMC5769823',
    'PMC5888214',
    'PMC5992089',
    'PMC6136514',
    'PMC6146047',
    'PMC6361124',
    'PMC6382152',
    'PMC6476335',
    'PMC6777629',
    'PMC6885166',
    'PMC6912774',
    'PMC6943000',
    'PMC7079832',
    'PMC7094873',
    'PMC7099239',
    'PMC7120385',
    'PMC7121408',
    'PMC7123078',
    'PMC7150046',
    'PMC7150310',
    'PMC7178827',
    'PMC7211701',
    'PMC7257586',
    'PMC7356254',
    'PMC7408088',
    'PMC7409586',
    'PMC7617339',
    'PMC7732539',
    'PMC7795544',
    'PMC7910878',
    'PMC8000082',
    'PMC8120005',
    'PMC8140133',
    'PMC8293617',
    'PMC8301367',
    'PMC8348155',
    'PMC8373144',
    'PMC8395727',
    'PMC8502845',
    'PMC8621025',
    'PMC8694271',
    'PMC8776198',
    'PMC8833991',
    'PMC88987',
    'PMC9107447',
    'PMC9122636',
    'PMC9161594',
    'PMC9193353',
    'PMC9254226',
    'PMC9279012',
    'PMC9316148',
    'PMC9351987',
    'PMC9428686',
    'PMC9445831',
    'PMC9464341',
    'PMC9482158',
    'PMC9585291',
    'PMC9616751',
    'PMC9758816',
    'PMC9760869',
    'PMC9818552',
    'PMC99140',
    'PMC9953674', 'PMC8930430', 'PMC8917243', 'PMC8728235', 'PMC8487971', 'PMC8728280',
    'PMC8465565', 'PMC8490017', 'PMC8361275', 'PMC8299054', 'PMC8305315',
    'PMC8274175', 'PMC8243463', 'PMC8245237', 'PMC8260255', 'PMC8221567',
    'PMC8293435', 'PMC8214732', 'PMC8199962', 'PMC8192694', 'PMC8098123',
    'PMC8225353', 'PMC8734201', 'PMC8067209', 'PMC8251562', 'PMC8039773',
    'PMC9455766', 'PMC7904003', 'PMC7915494', 'PMC7830961', 'PMC7828806',
    'PMC8456877', 'PMC8377004', 'PMC7820543', 'PMC7785925', 'PMC8247411',
    'PMC8583063', 'PMC7727275', 'PMC7708855', 'PMC7607678', 'PMC7693514',
    'PMC7855940', 'PMC7605523', 'PMC7579960', 'PMC8021618', 'PMC7508700',
    'PMC8247027', 'PMC7487458', 'PMC7450056', 'PMC7435196', 'PMC7460848',
    "PMC9817429", "PMC3501578", "PMC11062018", "PMC7607664", "PMC12004351", "PMC12117280",
    "PMC10258785", "PMC8799781", "PMC8282759", "PMC6371481", "PMC8783206", "PMC10114153",
    "PMC10568748", "PMC11707814", "PMC8461675", "PMC9859154", "PMC5822478", "PMC10880063",
    "PMC2396985", "PMC1183497", "PMC10474516", "PMC10393892", "PMC10234206", "PMC7755619",
    "PMC9351850", "PMC4045316", "PMC7510933", "PMC8094166", "PMC12141402", "PMC11253770",
    "PMC10496046", "PMC8094881", "PMC4874549", "PMC11827288", "PMC12162762", "PMC12102630",
    "PMC11816479", "PMC11771448", "PMC10530001", "PMC10255006", "PMC10403966", "PMC10330173",
    "PMC9976923", "PMC10721649", "PMC10467201", "PMC10083417", "PMC8743835", "PMC9662763",
    "PMC9521281", "PMC10613328", "PMC9303105", "PMC7821394", "PMC8923643", "PMC8347754",
    "PMC8092631", "PMC7922344", "PMC8035275", "PMC7611438", "PMC7882176", "PMC7044626",
    "PMC6663239", "PMC6373531", "PMC5889958", "PMC3514822", "PMC4997665", "PMC3132173",
    "PMC2718794", "PMC4820071", "PMC6923148", "PMC3595214", "PMC6483344", "PMC2745893",
    "PMC1694758", "PMC3321561", "PMC3310455", "PMC6147963", "PMC10837245", "PMC8959321",
    "PMC7127997", "PMC11610560", "PMC6250590", "PMC10527597", "PMC11988866", "PMC9797656",
    "PMC8942061", "PMC10875862", "PMC7748050", "PMC8094454", "PMC7240661", "PMC7198647",
    "PMC9351322", "PMC9553773", "PMC8156274", "PMC7454121", "PMC8006179", "PMC6221674",
    "PMC7169085", "PMC5567178", "PMC4280280", "PMC11078578", "PMC10486189", "PMC10090030",
    "PMC10081979", "PMC1349517", "PMC3309640", "PMC9583624", "PMC10084375", "PMC3310448",
    "PMC1349445", "PMC3647650", "PMC3358072", "PMC1446428", "PMC11230064", "PMC3358170",
    "PMC10591284", "PMC10492754", "PMC6122805", "PMC10600957", "PMC10280796", "PMC11025449",
    "PMC9251635", "PMC3298297", "PMC3557876", "PMC1508761", "PMC9616129", "PMC9527612",
    "PMC8205157", "PMC3647718", "PMC3414018", "PMC3559145", "PMC5754504", "PMC8321142",
    "PMC6362710", "PMC1694927", "PMC1404646", "PMC6470562", "PMC6016429", "PMC3322061",
    "PMC3294967", "PMC4889212", "PMC11890256", "PMC4822961", "PMC11426285", "PMC11136790",
    "PMC10949739", "PMC12143711", "PMC11709196", "PMC10531116", "PMC10247161", "PMC9997794",
    "PMC9981357", "PMC11660848", "PMC11620663", "PMC11596442", "PMC11463743", "PMC11432894",
    "PMC10898657", "PMC9772649", "PMC9388218", "PMC10187883", "PMC11272163", "PMC9169540",
    "PMC9038935", "PMC9014775", "PMC10601401", "PMC10349296", "PMC10313740", "PMC9969957",
    "PMC9951619", "PMC10511686", "PMC9264340", "PMC8587592", "PMC8587636", "PMC8329219",
    "PMC8503895", "PMC7851709", "PMC7779651", "PMC9783534", "PMC9629598", "PMC9985469",
    "PMC9675691", "PMC9232144", "PMC10126014", "PMC9544439", "PMC8813574", "PMC7587308",
    "PMC7499630", "PMC8188959", "PMC7255993", "PMC7279061", "PMC7613296", "PMC8524760",
    "PMC7050377", "PMC8805679", "PMC8586735", "PMC8440262", "PMC8314550", "PMC8259043",
    "PMC8234801", "PMC8245340", "PMC7983424", "PMC9118535", "PMC8092479", "PMC8096522",
    "PMC6751065", "PMC6699665", "PMC7795006", "PMC7784548", "PMC8231306", "PMC7332412",
    "PMC5501003", "PMC6517028", "PMC6198327", "PMC1112729", "PMC10419968", "PMC6687007",
    "PMC6686991", "PMC6377504", "PMC5946438", "PMC5901989", "PMC6502183", "PMC5623308",
    "PMC10174181", "PMC2598535", "PMC4208990", "PMC5470313", "PMC4986376", "PMC8713552",
    "PMC4768877", "PMC5210580", "PMC3658625", "PMC3359101", "PMC3785982", "PMC4826853",
    "PMC3457763", "PMC384330", "PMC4584373", "PMC5497591", "PMC3579756", "PMC4149319",
    "PMC5264294", "PMC4796376", "PMC5127176", "PMC3275800", "PMC2766390", "PMC2822988",
    "PMC3623661", "PMC6002842", "PMC1381161", "PMC4519429", "PMC3765736", "PMC2844366",
    "PMC4485633", "PMC3384378", "PMC6478277", "PMC3637496", "PMC7133546"
]

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = # Fill in your API key

# Obtain the full texr articles
def fetch_full_text(pmc_id):
  # send request to NCBI E-utilities API 
  fetch_url = f"{NCBI_BASE_URL}efetch.fcgi"

  # Define the fetch parameters
  params = {
        "db": "pmc",  # request for the PMC database
        "id": pmc_id,
        "retmode": "xml",  # obtain article in XML format
        "api_key": API_KEY
    }

  response = requests.get(fetch_url, params=params)
  response.raise_for_status()

  # return an XML response
  return response.text

def extract_text_from_xml(xml_data):
  # call Beautifulsoup to parse XML data to extract sections and text
  parse = BeautifulSoup(xml_data, 'xml')
  # extract all sections
  sections = parse.find_all('sec')

  # create an empty list for the extracted content
  content = []

  # Loop over each section
  for section in sections:
    # step 1: find Title and add to the content as header
    title = section.find('title')
    if title:
      content.append(f"\n##{title.get_text()}##\n" )

      # process text inside the sections
      for element in section.contents:
        # search for paragraph
        if element.name == 'p':
          content.append('\n' + element.get_text().strip() + '\n')

   # return the text
   return "\n".join(text_content)


# check full access for each PMC IDs (delete articles with restricted access)
def check_full_acces(xml_data):
  parser = BeautifulSoup(xml_data, 'xml')

# check for license information within the article
license = parser.find('license')

# Check if full body is present
if parser.find('body'):
  return True # Full text is available
return False # full text is not available

# save text in the corpus
def save_corpus(article_text, pmc_id):
  # create file with filename
  file = f"corpus/PMC_{pmc_id}.txt"
  
  # write text to the file
  with open(file, 'w', encoding='utf-8') as f:
    f.write(article_text)

# Clean heading text
def clean_heading_text(text):
    # Removes special characters in titles
    cleaned = re.sub(r'^#+\s*|\s*#+$', '', text.strip())
    return cleaned


# Main for running the functions
def main():
  for pmc_id in PMC_IDS:
    try:
      # obtain text from pmc ids
      xml_data = fetch_full_text(pmc_id)

      # skip restricted articles
      if not check_full_access(xml_data):
        continue

      # extract and save full article text
      article_text = extract_text_from_xml(xml_data)

      # remove white spaces and check for actual content
      if article_text.strip():
        # if text is found dave the text 
        save_to_file(article_text, pmc_id)

      # avoid hitting API limits
      time.sleep(1)

      # Load models for sentence splitting (SpaCy) and tokenization
      nlp = spacy.load("en_core_web_sm")  
      tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1") # ClinicalBERT: emilyalsentzer/Bio_ClinicalBERT
                                                                                    # BIOBERT; dmis-lab/biobert-base-cased-v1.1
                                                                                    # PubMedBERT: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract                                                                        
      
      # Path to corpus
      corpus_path = "/content/corpus_2"
      
      # create output file
      preprocessed_output_path = "preprocessed_corpus.json"  
      
      # create an empty list for processed text to store the data
      processed_sentences = []

      # Loop through the extracted files
      for files in os.listdir(corpus_path):
          if files.endswith(".txt"):  
              with open(os.path.join(corpus_path, files, "r", encoding="utf-8") as file:
                  text = file.read()
                
                  # split the sentences
                  doc = nlp(text)  
                
                  # Loop over the sentences
                  for sentence in doc.sents:
                      # clean the titles
                      cleaned_text = clean_heading_text(sentence.text)
                    
                      # Tokenize the text
                      tokens = tokenizer.tokenize(cleaned_text)  
      
                      # add special tokens for readability for the transformer models
                      tokens = ["[CLS]"] + tokens + ["[SEP]"] 
      
                      # add the preprocessed sentences to the list
                      processed_sentences.append(tokens)
      
      # Save  the preprocessed data to a JSON file
      with open(preprocessed_output_path, "w", encoding="utf-8") as f:
          json.dump(processed_sentences, f, indent=4)
        
# run the main function
if __name__ == "__main__":
  main()





