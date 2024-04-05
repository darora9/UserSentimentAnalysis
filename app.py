import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flair.models import TextClassifier
from flair.data import Sentence
from tqdm import tqdm


classifier = TextClassifier.load('en-sentiment')

def senti_score(n):
    s = Sentence(n)
    classifier.predict(s)
    total_sentiment = s.labels[0]
    assert total_sentiment.value in ["POSITIVE", "NEGATIVE"]
    sign = 1 if total_sentiment.value == "POSITIVE" else -1
    score = total_sentiment.score
    return sign * score

def sentiment_type(score):
    if score <= 0:
        return "Negative"
    elif score > 0:
        return "Positive"

import streamlit as st

st.title('User-Review Sentiment Analysis')
st.markdown('---')

inputMessage = st.text_area("Enter the text here")
if st.button('Predict'):
    score = senti_score(inputMessage)
    if sentiment_type(score) == 'Positive':
        st.header("Positive Review")
        st.write(f"Review Score is {round(score*2+2.5,2)}")
    else:
        st.header("Negative Review")
        st.write(f"Review Score is {round(score*2+2.5,2)}")

st.markdown("---")
st.header("Some Positive Samples:")
st.write("""
- With tiger 3 we have our perfect trilogy. Action with Emotions and story is the skeleton of this trilogy and Maneesh Sharma(dir) didn't disappoint. There were so many scenes, deja vu moments which reminded me of ett. TZH characters marked their presence beautifully. I got the content i craved for. Tiger3 is a perfect celebration feast for tiger zoya fans. Salman Khan lives tiger. He walks, talks and breathes tiger. His one stare, his subtle dialogue delivery, his chemistry with Katrina, everything was pitch perfect. His stint as tiger would always be my favourite. His body stature is made for those epic back poses. His presence Is god level. He just has to be there on my screen, and my eyes lit up. Emraan Hashmi cake walks. He's simply brilliant. He's effective and scary. Not that he has got unreal weapons or something, but Emraan's aatish was effortlessly menacing. His voice and walk >> Katrina Kaif proves why everyone calls her mother of spy universe. She was the best presented in this movie. Her character add-ons were so good that nothing felt forced or out of place. RUAAN RUAAN IS NEXT BEST OR EQUALLY BEST AS SAIYAARA. That song, the setting, trust me. Makers saved it for the best. Experience it in theatres yourself.
- A wonderful little production. The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. The actors are extremely well chosen- Michael Sheen not only 'has got all the polari' but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master's of comedy and his life. The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional 'dream' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals decorating every surface) are terribly well done.
""")

st.header("Some Negative Samples:")
st.write("""
- Traumatized after watching this movie, slept twice in the middle of the movie 😞 The action scenes with body double and slow motion looked like THUMBS UP commercial The review that you see here with five stars are written by Salman Khan himself Most movie is shot on Green scene as Salman Khan is too old to perform his own stunts But he is still doing lunges and calls it dance moves its puzzling why filmmakers invest so much these days in deceiving people everything is poorly executed especially the portrayal of the lead character, which is quite pathetic It's unclear when they will retire old talent and bring in fresh faces from India Several companies openly engage in this practice Just wait a couple of months and you will see the ratings and reviews from real users which will likely drop this movie to less than two stars with numerous negative reviews Liger was a better movies than this tiger 🤣 The plot seems convoluted and lacks coherence jumping from one subplot to another without proper development We can see some vfx issues during fight scenes Overall nothing new Some scene are okay but overall not that much good They should have focused more in the story rather than fighting scene Expected more from Imran and Katrina But didn't get enough chances to make noticable impacts
- The most overblown tosh I've seen in years. An hysterical orgy of jump cuts almost designed to obscure the narrative. Endless shots of obsessive indecipherable scrawling on blackboards. Constant employment of that most irritating and hackneyed device, deep discussion on the move. Fabulously distracting use of flashes of fire, electricity, explosions and splitting of the atom throughout: random and unilluminating. Who are these characters? What are their relationships? Up pops the usually dependable Florence Pugh replaced as love interest moments later by Emily Blunt, also a habitually strong turn. By turns black and white and colour, the narrative felt wilfully confusing. Tom Conti’s hammy Einstein was a welcome relief from the endless parade of indistinguishable scientists in suits storming in and out of different, ill explained rooms. Mark Damon exploded occasionally over nothing more than the requirement to provide drama. Awful dramatic music, completely unconnected to the pace of events depicted, signalled an extraordinary event at any moment. These failed to materialise. Nuclear fission is a fairly complex concept for all but the experts I would suggest. This film did nothing to illuminate it for a wider audience. Tricksy, shallow and 3 frikkin hours of it. We ran out half way through to conduct a successful dry martini experiment at home.
""")
