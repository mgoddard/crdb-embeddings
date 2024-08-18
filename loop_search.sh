#!/bin/bash

queries=( "How to pack light for a trip" "in 2009 Moeller signed a minor league contract with the Baltimore Orioles" "which amplifiers have vacuum tube triode sound" "Shelby sends Phil Hill and Bruce McLaren to Le Mans" "Its enchanting coastline is all breathtaking panoramas, idyllic beaches and pools" "It starts with ginger-scented chicken-cilantro meatballs that are browned" "two distinct deserts—the Mojave and Sonoran—as it tumbles down from the heights into the Coachella Valley near Palm Springs" "The park embraces parts of two distinct deserts—the Mojave and Sonoran" "breaks the weak isospin symmetry of the electroweak interaction" )

for n in {1..10000}
do
  q=${queries[ $RANDOM % ${#queries[@]} ]}
  echo "$q"
  time ./search_client.sh $q
done

