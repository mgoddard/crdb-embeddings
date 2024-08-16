#!/usr/bin/env perl

use strict;

# 2021-07-08 12:36:45<Bellairs<'''Bellairs''' and '''Bellair''' are surnames. Notable people with the surnames include: * [[Angus Bellairs]] (1918–1990), British herpetologist and anatomist * [[Bart Bellairs]] (born 1956), American basketball coach * [[Carlyon Bellairs]] (1871–1955), British Royal Navy officer * [[Edmund Bellairs]] (1823–1898), New Zealand legislator * [[George Bellairs]]

my $MIN_TEXT_LEN = 256;

my %ent = (
  "lt" => '<'
  , "gt" => '>'
  , "quot" => '"'
  , "amp" => '&'
  , "nbsp" => ' '
);
my $re = join('|', keys %ent);

while (<>)
{
  s/'{2,3}/"/g;
  s/\[\[//g;
  s/\]\]//g;
  my ($dt, $title, $text) = split /</;
  next if $text =~ /(#REDIRECT|(may|can)\s+refer\s+to[:\s+])/i;
  # Title
  $title = lc($title);
  $title =~ s/\s+$//;
  $title =~ s/^\s+//;
  $title =~ s/\s+/_/g;
  $title =~ s/\W+//;
  $title =~ s~[\.,\(\)/']~~g;
  # Text
  $text =~ s/&($re);/$ent{$1}/g;
  $text =~ s~<ref[^/]+/>~~g;
  $text =~ s~<ref[^>]*>~~g;
  $text =~ s~</ref>~~g;
  $text =~ s~\[.+?\]~~g;
  $text =~ s~{{.+?}}~~g;
  $text =~ s~===.+?===~~g;
  next unless length($text) >= $MIN_TEXT_LEN;
  #print "title: $title, text: $text\n";
  print "title: $title\n";
  open OUT, "> ./wiki_pages/$title.txt" or die $!;
  print OUT $text;
  close OUT or warn $!;
}

