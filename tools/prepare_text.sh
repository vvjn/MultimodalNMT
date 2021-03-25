#!/bin/bash

BPE=.
MOSES_PATH=moses-5cbafabfd/tokenizer
PATH=${MOSES_PATH}:$PATH
SUFF="norm.tok.lc"

BPE_OPS=10000

for tlang in de; do
  echo "Preparing en-${tlang} dataset"
  folder="en-${tlang}"
  mkdir -p $folder
  for sp in train val test; do
    # Process both sides
    for llang in en ${tlang}; do
      inp="raw/${sp}.${llang}"
      if [ -f $inp ]; then
        cat $inp | lowercase.perl -l ${llang} | normalize-punctuation.perl -l ${llang} | \
          tokenizer.perl -l ${llang} -a -threads 4 > $folder/${sp}.${SUFF}.${llang}
      fi
    done

    trg="${sp}.${SUFF}.${tlang}"

    # De-hyphenize test set targets for proper evaluation afterwards
    if [[ "$sp" =~ ^test.* ]] && [[ -f "${folder}/${trg}" ]]; then
      sed -r 's/\s*@-@\s*/-/g' < ${folder}/${trg} > ${folder}/${trg}.dehyph
    fi
  done

  for llang in en ${tlang}; do
      python $BPE/learn_bpe.py -s $BPE_OPS < $folder/train.${SUFF}.${llang} > $folder/bpe-codes.${llang}

      for sp in train val test; do
          inp="raw/${sp}.${llang}"
          if [ -f $inp ]; then
              python $BPE/apply_bpe.py -c $folder/bpe-codes.${llang} < $folder/${sp}.${SUFF}.${llang} > $folder/${sp}.${SUFF}.bpe${BPE_OPS}.${llang}
          fi
      done
  done
done
