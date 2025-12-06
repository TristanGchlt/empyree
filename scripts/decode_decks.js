// scripts/decode_decks.js
import fs from 'fs';
import alteredDeckfmt from 'altered-deckfmt';
const { decodeList } = alteredDeckfmt;
import { parse } from 'csv-parse/sync';
import { stringify } from 'csv-stringify/sync';

const inputCsv = 'data/raw/extract_2025_12_04.csv';
const outputCsv = 'data/decoded/decoded_2025_12_04.csv';

const raw = fs.readFileSync(inputCsv, 'utf-8');
const records = parse(raw, { columns: true, skip_empty_lines: true });

const decoded = records.map(r => {
  try {
    const deck = decodeList(r.deckCode);
    return { ...r, cards: JSON.stringify(deck) };
  } catch (e) {
    return { ...r, cards: null };
  }
});

const valid = decoded.filter(r => r.cards && r.cards.length > 0);

fs.mkdirSync('data/decoded', { recursive: true });
const out = stringify(valid, { header: true, columns: Object.keys(valid[0]) });
fs.writeFileSync(outputCsv, out);

console.log(`Décodé ${valid.length} decks dans ${outputCsv}`);