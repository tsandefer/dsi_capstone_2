# Take JSON file, import select records
import json, os, requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup # to parse html
from operator import itemgetter
from pprint import pprint
import lyricsgenius

GENIUS_CLIENT_ACCESS_TOKEN = os.environ['GENIUS_CLIENT_ACCESS_TOKEN']

defaults = {
    'request': {
        'token': GENIUS_CLIENT_ACCESS_TOKEN,
        'base_url': 'https://api.genius.com'
    },
    'message': {
        'search_fail': 'The lyrics for this song were not found!',
        'wrong_input': 'Wrong number of arguments.\n' \
                       'Use two parameters to perform a custom search ' \
                       'or none to get the song currently playing on Spotify.'
    }
}


artist_titles_dict = {'Lil Wayne': array(['A Milli', 'Right Above It', '6 Foot 7 Foot', 'Drop the World',
        'She Will', 'Mirror', "Blunt Blowin'", 'No Worries', 'Love Me',
        'Rich As Fuck', 'Believe Me', "Don't Cry", 'Uproar', 'Mona Lisa']),
 'Eminem': array(['Space Bound', "Cleanin' Out My Closet", 'Superman',
        'Like Toy Soldiers', 'Sing for the Moment', 'No Love', 'Beautiful',
        '8 Mile: B-Rabbit vs Papa Doc', "When I'm Gone", 'Not Afraid',
        'Mockingbird', "'Till I Collapse", 'Lose Yourself',
        'The Real Slim Shady', 'The Way I Am', 'Survival', 'Berzerk',
        'Without Me', 'The Monster', 'Stronger Than I Was', 'Love Game',
        'Headlights', 'Bad Guy', 'Legacy', 'Evil Twin', 'Rap God', 'Stan',
        'My Name Is', 'Love the Way You Lie', 'Guts Over Fear',
        'Detroit vs. Everybody', 'Walk on Water', 'Untouchable', 'River',
        'Believe', 'Fall', 'Greatest',
        'Venom (Music from the Motion Picture)', 'Not Alike', 'Kamikaze',
        'The Ringer', 'Lucky You', 'Killshot']),
 'J. Cole': array(['In the Morning', 'Lost Ones', 'Work Out', "Can't Get Enough",
        "Nobody's Perfect", 'Power Trip', 'Crooked Smile', 'Let Nas Down',
        'Villuminati', 'LAnd of the Snakes', 'Forbidden Fruit', 'Trouble',
        'She Knows', 'Born Sinner', 'Rich Niggaz', 'Sparks Will Fly',
        'Runaway', 'Wet Dreamz', 'January 28th', "03' Adolescence",
        'No Role Modelz', 'Love Yourz', 'Apparently', 'G.O.M.D.',
        'Fire Squad', 'A Tale of 2 Citiez', 'Black Friday',
        'False Prophets', '\u200beverybody dies',
        'For Whom the Bell Tolls', "She's Mine, Pt. 1", 'Deja Vu',
        'Neighbors', "She's Mine, Pt. 2", 'Immortal', 'Change',
        '4 Your Eyez Only', 'KOD', 'Photograph', 'The Cut Off', 'Motiv8',
        'ATM', '1985 (Intro to "The Fall Off")', 'FRIENDS',
        "Kevin's Heart", 'MIDDLE CHILD']),
 'JAY-Z': array(['Run This Town', 'Renegade', 'Open Letter', 'Holy Grail', 'Oceans',
        'Heaven', 'FuckWithMeYouKnowIGotIt', 'Picasso Baby', 'Tom Ford',
        'Part II (On The Run)', '99 Problems', 'Empire State of Mind',
        'Takeover', 'Song Cry', 'Kill Jay Z', 'The Story of O.J.', '4:44',
        'Family Feud']),
 'Kanye West': array(['POWER', "Can't Tell Me Nothing", 'Dark Fantasy', 'Runaway',
        'Gorgeous', "Don't Like.1", 'Mercy', 'Clique', 'Monster',
        'Gold Digger', 'Black Skinhead', 'New Slaves', 'Blame Game',
        'Blood on the Leaves', 'Bound 2', 'Hold My Liquor', "I'm In It",
        'I Am a God', 'Only One', 'All Day',
        'Facts (Charlie Heat Version)', 'Real Friends',
        'No More Parties in\xa0LA', 'Father Stretch My Hands, Pt. 1',
        'FML', 'Ultralight Beam', 'Wolves', 'Famous', 'Pt. 2',
        'I Love Kanye', 'Waves', 'Saint Pablo', 'Lift Yourself',
        'All Mine', 'Ghost Town']),
 'The Weeknd': array(['Wicked Games', 'The Morning', 'Rolling Stone',
        'The Party & The After Party', 'Loft Music', 'The Zone',
        'Initiation', 'Montreal', 'Twenty Eight', 'Kiss Land',
        'Drunk In Love (Remix)', 'Often', 'King of the Fall', 'Earned It',
        'The Hills', "Can't Feel My Face", 'In the Night', 'Prisoner',
        'Acquainted', 'Dark Times', 'Angel', 'Shameless',
        'Tell Your Friends', 'Starboy', 'Party Monster',
        'I Feel It Coming', 'Six Feet Under', 'Sidewalks', 'Die for You',
        'Reminder', 'Try Me', 'Call Out My Name', 'Wasted Times',
        'Privilege']),
 'Drake': array(["Marvin's Room", 'Shot for Me', 'HYFR', 'Over My Dead Body',
        'The Motto', 'Started from the Bottom', 'Girls Love Beyoncé',
        'All Me', "Hold On, We're Going Home", 'Too Much',
        'Pound Cake / Paris Morton Music 2', 'Worst Behavior', 'From Time',
        'The Language', 'Furthest Thing', '0 to 100 / The Catch Up',
        'How About Now', 'Jungle', 'Legend', 'Energy', 'Know Yourself',
        'Hotline Bling', 'Back to Back', 'Summer Sixteen', 'One Dance',
        "Child's Play", 'Too Good', 'Fake Love', 'Passionfruit',
        'Portland', "God's Plan", 'Nice For What', 'In My Feelings']),
 'Kendrick Lamar': array(['A.D.H.D', 'HiiiPoWeR', 'The Recipe', 'Swimming Pools (Drank)',
        'Backseat Freestyle', 'The Art of Peer Pressure',
        "Bitch, Don't Kill My Vibe", "Sing About Me, I'm Dying of Thirst",
        'Poetic Justice', 'Money Trees',
        "Sherane a.k.a Master Splinter's Daughter", '\u200bm.A.A.d city',
        'The Blacker the Berry', 'King Kunta', "Wesley's Theory",
        'Mortal Man', 'How Much a Dollar Cost', '\u200bu', 'Alright',
        'Hood Politics', 'These Walls', '\u200bi (Album Version)',
        'Black Friday', 'The Heart Part 4', 'HUMBLE.', 'DUCKWORTH.',
        'LOYALTY.', 'ELEMENT.', 'XXX.', 'LOVE.', 'DNA.', 'FEEL.', 'PRIDE.',
        'Rigamortus', 'FEAR.']),
 'Ariana Grande': array(['The Way', 'Right There', 'Almost Is Never Enough', 'Problem',
        'Break Free', 'Best Mistake', 'Love Me Harder', 'One Last Time',
        'Focus', 'Dangerous Woman', 'Be Alright', 'Let Me Love You',
        'Into You', 'Everyday', 'Side to Side', 'Moonlight',
        '\u200b\u200bno tears left to cry', '\u200bthe light is coming',
        '\u200bGod is a woman', '\u200bbetter off', '\u200bget well soon',
        '\u200bR.E.M', '\u200bpete davidson', '\u200bsweetener',
        '\u200b\u200bbreathin', '\u200braindrops (an angel cried)',
        '\u200bsuccessful', '\u200beverytime', '\u200bgoodnight n go',
        '\u200bthank u, next', '\u200bneedy', '\u200bimagine', '7 rings',
        '\u200bfake smile', '\u200bmake up', 'NASA',
        "\u200bbreak up with your girlfriend, i'm bored",
        '\u200bin my head', '\u200bbloodline', '\u200bghostin',
        '\u200bbad idea']),
 'Beyoncé': array(['Blow', 'Mine', '***Flawless', 'Drunk in Love', 'Partition',
        'Rocket', '***Flawless (Remix)', 'FORMATION', 'HOLD UP',
        'PRAY YOU CATCH ME', 'ALL NIGHT', 'SORRY']),
 'Original Broadway Cast of Hamilton': array(['The Story of Tonight', 'What Comes Next?',
        'Best of Wives and Best of Women', "A Winter's Ball",
        'Dear Theodosia', 'History Has Its Eyes on You',
        'Alexander Hamilton', 'Schuyler Defeated', 'I Know Him',
        'The Adams Administration', 'Stay Alive (Reprise)', 'Stay Alive',
        'Guns and Ships', 'Aaron Burr, Sir', 'The Reynolds Pamphlet',
        "What'd I Miss", 'Yorktown (The World Turned Upside Down)', 'Burn',
        'We Know', "You'll Be Back", 'Hurricane', 'Cabinet Battle #1',
        'Take a Break', 'Meet Me Inside', 'Washington on Your Side',
        'Who Lives, Who Dies, Who Tells Your Story', 'Helpless',
        "It's Quiet Uptown", 'Ten Duel Commandments', 'My Shot',
        'The Story of Tonight (Reprise)', 'Farmer Refuted',
        'Your Obedient Servant', 'One Last Time', 'That Would Be Enough',
        'The World Was Wide Enough', 'Blow Us All Away', 'Satisfied',
        'Cabinet Battle #2', 'Right Hand Man', 'Wait for It', 'Non-Stop',
        'The Schuyler Sisters', 'Say No to This',
        'The Room Where It Happens', 'The Election of 1800']),
 'XXXTENTACION': array(["LET'S PRETEND WE'RE NUMB", 'YuNg BrAtZ',
        "I don't wanna do this anymore", '#ImSippinTeaInYoHood',
        'WingRiddenAngel', 'RIOT', 'ILOVEITWHENTHEYRUN', 'Look At Me!',
        'RIP Roach (East Side Soulja)', 'King of the Dead', 'KING',
        'I spoke to the devil in Miami, he said everything would be fine',
        'Revenge', 'Vice City', 'XXL Freshman Freestyle: XXXTENTACION',
        'Everybody Dies in Their Nightmares', 'Fuck Love', 'Carry On',
        'Jocelyn Flores', 'Depression & Obsession', 'Save Me', 'Orlando',
        'Dead Inside (Interlude)', 'Ayala (Outro)',
        'A GHETTO CHRISTMAS CAROL', 'UP LIKE AN INSOMNIAC (Freestyle)',
        'Hope', '\u200bchanges', 'SAD!', 'ALONE, PART 3',
        '\u200bthe remedy for a broken heart (why am I so in love)',
        'Moonlight', 'NUMB', '\u200binfinity (888)',
        "I don't even speak spanish lol", 'BAD!', 'Guardian angel',
        'Train food'])}

genius = lyricsgenius.Genius(GENIUS_CLIENT_ACCESS_TOKEN, sleep_time=0, verbose=False, remove_section_headers=True)
# lyrics = genius._scrape_song_lyrics_from_url()

full_lyrics_dict = dict()

for artist_name in artist_titles_dict.keys():
    for song_title in artist_titles_dict[artist_name]:
        song = genius.search_song(song_title, artist_name)
        lyrics = song.lyrics
        full_lyrics_dict[song_title] = lyrics


        # lines = lyrics.split('\n')
        # for line in lines:
        #      if len(line) == 0:
        #          lines.remove(line)
        # full_lyrics = lines
