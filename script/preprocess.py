#!/usr/bin/env python
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

class CDRProcessor:
    def parse_bioc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        samples = []
        
        for doc in root.findall('.//document'):
            # Concatenate passage texts into a single document text
            text_parts = []
            for p in doc.findall('passage'):
                if p.find('text') is not None and p.find('text').text:
                    text_parts.append(p.find('text').text)
            text = ' '.join(text_parts)

            # Extract Chemical and Disease entities with their MESH IDs
            entities = defaultdict(list)
            for ann in doc.findall('.//annotation'):
                ent_type = ann.find("infon[@key='type']").text
                mesh_id = ann.find("infon[@key='MESH']").text if ann.find("infon[@key='MESH']") is not None else None
                text_span = ann.find('text').text
                if ent_type in ['Chemical', 'Disease'] and mesh_id:
                    entities[ent_type].append((text_span, mesh_id))

            # Extract chemical-disease relations labeled 'CID'
            relations = set()
            for rel in doc.findall('.//relation'):
                relation_type = rel.find("infon[@key='relation']")
                if relation_type is not None and relation_type.text == 'CID':
                    chem = rel.find("infon[@key='Chemical']").text if rel.find("infon[@key='Chemical']") is not None else None
                    dis = rel.find("infon[@key='Disease']").text if rel.find("infon[@key='Disease']") is not None else None
                    if chem and dis:
                        relations.add((chem, dis))

            # Generate sample records for every Chemical-Disease pair
            for chem, chem_id in entities.get('Chemical', []):
                for dis, dis_id in entities.get('Disease', []):
                    label = 1 if (chem_id, dis_id) in relations else 0
                    samples.append({
                        'text': text,
                        'chemical': chem,
                        'disease': dis,
                        'label': label,
                        'chem_id': chem_id,
                        'dis_id': dis_id
                    })
        return samples


def main():
    # Path configuration
    data_dir = Path("EAAE/data")
    output_dir = Path("EAAE/data/processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = CDRProcessor()
    
    # Process train, dev, and test splits
    splits = {
        'train': 'CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml',
        'dev': 'CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml',
        'test': 'CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml'
    }
    
    for split, rel_path in splits.items():
        samples = processor.parse_bioc_xml(data_dir / rel_path)
        
        # Save processed samples to JSON
        with open(output_dir / f"{split}.json", 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        # Print dataset statistics for each split
        pos_count = sum(s['label'] for s in samples)
        total = len(samples)
        neg_count = total - pos_count
        print(f"{split.upper():<6} | Total: {total:<5} | Pos: {pos_count:<4} | Neg: {neg_count:<4} | Pos%: {pos_count/total:.1%}")

if __name__ == '__main__':
    main()
