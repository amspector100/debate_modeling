# Debate modeling

This repository is the code I used to create some simple models of debating results.

## Data availability

To make things easy, I have pushed all of the raw data to github. You can simply navigate to ``scrape/`` and run ``python3.9 combine.py`` to clean/combine all of this data, which will then be ready for analysis.

The combined data will appear in the ``data/combined/`` directory. There will be three files: (i) ``all_speakers.csv`` which is the list of all speakers and their unique speaker id, ``all_speaks.csv`` which is the list of all speaks obtained by speakers, and ``round_data.csv``, which lists all results from all rounds in the tournaments (including both inrounds and outrounds).

## Scraping new data

The file ``data/tournaments.csv`` lists the tournaments in the current dataset. If you fork this repository and would like to add additional tournaments, just add additional lines to this CSV (make sure to give every tournament a unique tournament ID).

Once you've done this, navigate to the ``scrape/`` directory and run ``python3.9 scrape.py`` to scrape the data from calicotab. You will need to install/configure Selenium for this because I was lazy and couldn't figure out a better way to do it.

## Modelling

I have made a few preliminary models, some of which are available in ``notebooks_public/``. If you train my best model only on inrounds, it achieves roughly 68% accuracy in predicting who will advance in outrounds. I don't know if this is good (I'm not even sure it's an interesting prediction task), but I haven't had time to do anything better... :).

## Simulations

The folder ``sims/`` contains the code to simulate some tournaments. I ran some experiments to see if tapered scoring is a good idea---I personally found that tapering/not tapering doesn't make much difference, but maybe in your simulations you'll find different results.