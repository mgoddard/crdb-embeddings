#!/bin/bash

# /home/mgoddard/sst/mac_mgoddard/CockroachDB/BaT_data

links -dump $1 | perl -e '$ok = 0; %seen = (); @lines = (); while (<>) { if (not $ok) { if (/Advertise on BaT/) { $ok = 1; } next; } chomp; s/^\s+//g; last if /(IFrame|Photo Gallery)/; push(@lines, $_) if not $seen{$_}; $seen{$_} += 1 if length($_) > 4; } $max_k = undef; $max = 0; foreach $k (keys %seen) { if ($seen{$k} > $max) { $max = $seen{$k}; $max_k = $k; } }; $max_k = lc($max_k); $max_k =~ s/^\s*No Reserve: *//g; $max_k =~ s/ +/_/g; $max_k =~ s/[^\w_]+//g; open OUT, "> ./bat/${max_k}.txt" or die $!; print OUT join("\n", @lines) . "\n"; close OUT;'

