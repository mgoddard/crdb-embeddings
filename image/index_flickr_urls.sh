#!/bin/bash

if [[ "$#" -ne 1 ]]
then
  echo "Usage: $0 offset_value (e.g. 300)"
  exit 1
fi

offset=$1
limit=1000

for url in $( tail -n +$(( offset + 1 )) ./flickr_urls.txt | head -$limit )
do
  time ./index_image.py $url
done

echo
echo "NEXT OFFSET: $(( offset + limit ))"
echo
