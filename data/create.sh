# this script downloads and extracts the
# nap, vec, pms, lmo, scn, eml, lld, lij, sc, fur, roa_tara
# wikipedias from 01.03.2022

wget https://dumps.wikimedia.org/astwiki/20231201/astwiki-20231201-pages-articles-multistream.xml.bz2
wget https://dumps.wikimedia.org/euwiki/20231201/euwiki-20231201-pages-articles-multistream.xml.bz2
wget https://dumps.wikimedia.org/glwiki/20231201/glwiki-20231201-pages-articles-multistream.xml.bz2
wget https://dumps.wikimedia.org/anwiki/20231201/anwiki-20231201-pages-articles-multistream.xml.bz2
wget https://dumps.wikimedia.org/ladwiki/20231201/ladwiki-20231201-pages-articles-multistream.xml.bz2
#wget https://dumps.wikimedia.org/cawiki/20231201/cawiki-20231201-pages-articles-multistream.xml.bz2
#wget https://dumps.wikimedia.org/eswiki/20231201/eswiki-20231201-pages-articles-multistream.xml.bz2


python -m wikiextractor.WikiExtractor astwiki-20231201-pages-articles-multistream.xml.bz2 -o ast_texts --json
python -m wikiextractor.WikiExtractor euwiki-20231201-pages-articles-multistream.xml.bz2 -o eu_texts --json
python -m wikiextractor.WikiExtractor glwiki-20231201-pages-articles-multistream.xml.bz2 -o gl_texts --json
python -m wikiextractor.WikiExtractor anwiki-20231201-pages-articles-multistream.xml.bz2 -o an_texts --json
python -m wikiextractor.WikiExtractor ladwiki-20231201-pages-articles-multistream.xml.bz2 -o lad_texts --json
#python -m wikiextractor.WikiExtractor cawiki-20231201-pages-articles-multistream.xml.bz2 -o ca_texts --json
#python -m wikiextractor.WikiExtractor eswiki-20231201-pages-articles-multistream.xml.bz2 -o es_texts --json





python generation.py

#rm -r *_texts
rm *.xml.bz2