import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = []
labels = []
texts_pos = []
texts_neg = []
# open files
with open('rt-polarity.pos', 'r', encoding='utf-8') as file:
    # read line by line
    while True:
        line = file.readline()
        texts.append(line)
        labels.append(1)
        texts_pos.append(line)

        if not line:
            break

with open('rt-polarity.neg', 'r', encoding='utf-8') as file:
    # read line by line
    while True:
        line = file.readline()
        texts.append(line)
        labels.append(0)
        texts_neg.append(line)
   
        if not line:
            break

# traing sets and test sets: 8:2
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

import random

# take 100 distinct numbers in [1, 5000]
random_numbers = random.sample(range(1, 5001), 100)

# positive sentiment text
text_pos = ""
for i in range(100):
    text_pos+=texts[random_numbers[i]]
    # text_pos.append(texts[random_numbers[i]])

# neutral sentiment text
text_normal = ""
for i in range(50):
    text_normal += texts[random_numbers[i]]
    text_normal += texts[len(texts)-random_numbers[i]-1]

# negative sentiment text
text_neg = ""
for i in range(100):
    text_neg += texts[len(texts)-random_numbers[i]-1]

# text_pos_vec = []
# for i in range(100):
#     random_numbers = random.sample(range(1, 5001), 50)
#
# text_neg_vec = []
# for i in range(100):
#     random_numbers = random.sample(range(1, 5001), 50)

def drawPlot(data):
    x = np.arange(0, len(data))
    y = np.array(data)
    window_size = 3

    # moving average
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    # adjust x to ensure len(x) = len(y_smooth)
    x_smooth = x[(window_size-1)//2 : -(window_size-1)//2]


    plt.plot(x, y, label='root')
    plt.plot(x_smooth, y_smooth, label='smooth')
    plt.title('plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def preprocess(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return inputs


def preprocess_t(texts, labels):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    labels = torch.tensor(labels)
    return inputs, labels


# predict function 
def predict_sentiment(text):
    # model.eval()
    inputs = preprocess([text])
    # with torch.no_grad():
    outputs = model(**inputs)
    fuck = outputs.logits
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return 1 if prediction == 1 else -1

def predict_label(sentence):
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)[0]
    # get the label with the highest mark
    predicted_label = torch.argmax(outputs, dim=1).item()
    # get the label name
    labels = model.config.id2label.values()
    predicted_class = list(labels)[predicted_label]
    return predicted_class

test_inputs, test_labels = preprocess_t(test_texts, test_labels)

model_acc_data = []

for i in range(20):
    file_name = str(i) + '_model.pth'
    model = torch.load(file_name)
    # evaluate the model
    model.eval()
    with torch.no_grad():
        output = model(**test_inputs)
        predictions = torch.argmax(output.logits, dim=1)
        accuracy = (predictions == test_labels).numpy().mean()
        model_acc_data.append(accuracy)
        print(f"Accuracy: {accuracy}")


plt.plot(model_acc_data)
plt.title('Line Chart Example')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
text = "Every day is a new beginning. Take a deep breath, smile, and start again.  Success is not final, failure is not fatal: It is the courage to continue that counts. The best time to plant a tree was 20 years ago.  The second best time is now. Believe you can and you're halfway there. Your positive action combined with positive thinking results in success. Happiness is not something ready-made.  It comes from your own actions. With the new day comes new strength and new thoughts. You are never too old to set another goal or to dream a new dream. The only way to do great work is to love what you do. It does not matter how slowly you go as long as you do not stop. Our greatest glory is not in never falling, but in rising every time we fall. All our dreams can come true, if we have the courage to pursue them. It always seems impossible until it's done. The power of imagination makes us infinite. Change your thoughts and you change your world. Happiness is not by chance, but by choice. Life is 10% what happens to us and 90% how we react to it. The best preparation for tomorrow is doing your best today. The only limit to our realization of tomorrow will be our doubts of today. What you get by achieving your goals is not as important as what you become by achieving your goals. Life is a succession of lessons which must be lived to be understood. You don’t have to be great to start, but you have to start to be great. Always do your best.  What you plant now, you will harvest later. Be not afraid of life.  Believe that life is worth living, and your belief will help create the fact. To accomplish great things, we must not only act, but also dream; not only plan, but also believe. Our lives begin to end the day we become silent about things that matter. Do not wait; the time will never be 'just right. ' Start where you stand, and work with whatever tools you may have at your command, and better tools will be found as you go along. Set your goals high, and don’t stop till you get there. Be kind whenever possible.  It is always possible. Quality is not an act, it is a habit. The only source of knowledge is experience. A truly rich man is one whose children run into his arms when his hands are empty. A room without books is like a body without a soul. We become what we think about. An unexamined life is not worth living. Eighty percent of success is showing up. Your time is limited, don’t waste it living someone else’s life. Winning isn’t everything, but wanting to win is. I am not a product of my circumstances.  I am a product of my decisions. Either you run the day, or the day runs you. Whether you think you can or you think you can’t, you’re right. The two most important days in your life are the day you are born and the day you find out why. Whatever you can do, or dream you can, begin it.  Boldness has genius, power, and magic in it. The best revenge is massive success. People often say that motivation doesn’t last.  Well, neither does bathing.  That’s why we recommend it daily. Life shrinks or expands in proportion to one's courage. If you hear a voice within you say 'you cannot paint,' then by all means paint and that voice will be silenced. There is only one way to avoid criticism: do nothing, say nothing, and be nothing. Ask and it will be given to you; search, and you will find; knock and the door will be opened for you. The only person you are destined to become is the person you decide to be. Go confidently in the direction of your dreams.  Live the life you have imagined. When I let go of what I am, I become what I might be. Life is not measured by the number of breaths we take, but by the moments that take our breath away. Happiness is not something you postpone for the future; it is something you design for the present. Remember that not getting what you want is sometimes a wonderful stroke of luck. You can’t use up creativity.  The more you use, the more you have. Dream big and dare to fail. Our greatest weakness lies in giving up.  The most certain way to succeed is always to try just one more time. The mind is everything.  What you think you become. I would rather die of passion than of boredom. A truly rich man is one whose children run into his arms when his hands are empty. It is never too late to be what you might have been. Build your own dreams, or someone else will hire you to build theirs. Education costs money.  But then so does ignorance. I have learned over the years that when one's mind is made up, this diminishes fear. It does not matter how slowly you go, so long as you do not stop. Limitations live only in our minds.  But if we use our imaginations, our possibilities become limitless. You become what you believe. I would rather die of passion than of boredom. A person who never made a mistake never tried anything new. What’s money? A man is a success if he gets up in the morning and goes to bed at night and in between does what he wants to do. I didn’t fail the test.  I just found 100 ways to do it wrong. In order to succeed, your desire for success should be greater than your fear of failure. A person who never made a mistake never tried anything new. The only person who is educated is the one who has learned how to learn and change. There are no traffic jams along the extra mile. It is never too late to be what you might have been. The only way to do great work is to love what you do. If you can dream it, you can achieve it. Remember no one can make you feel inferior without your consent. Life is what we make it, always has been, always will be. The question isn’t who is going to let me; it’s who is going to stop me. When everything seems to be going against you, remember that the airplane takes off against the wind, not with it. It’s not the years in your life that count.  It’s the life in your years. Change your thoughts and you change your world. Either write something worth reading or do something worth writing. Nothing is impossible, the word itself says, “I’m possible!”The only way to do great work is to love what you do. If you can dream it, you can achieve it. The limit to your abilities is where you place it. Being happy doesn't mean that everything is perfect.  It means that you've decided to look beyond the imperfections. Opportunities don't happen, you create them. Love the life you live.  Live the life you love. To live a creative life, we must lose our fear of being wrong. Do not let what you cannot do interfere with what you can do. Life is not about finding yourself.  Life is about creating yourself. To handle yourself, use your head; to handle others, use your heart. Too many of us are not living our dreams because we are living our fears. Do not go where the path may lead, go instead where there is no path and leave a trail. Life is 10% what happens to me and 90% of how I react to it. "

def group_sentences(text, n):
    sentences = text.split(" ")
    # sentences = text.split("。")  # split the text by 'period'
    grouped_sentences = []
    for i in range(0, len(sentences), n):
        grouped_sentence = ' '.join(sentences[i:i+n])  # reconnect the fragmented sentences
        grouped_sentences.append(grouped_sentence)
    return grouped_sentences
text = "Every day is a new beginning. Take a deep breath, smile, and start again.Success is not final, failure is not fatal: It is the courage to continue that counts.The best time to plant a tree was 20 years ago. The second best time is now.Believe you can and you're halfway there.Your positive action combined with positive thinking results in success.Happiness is not something ready-made. It comes from your own actions.With the new day comes new strength and new thoughts.You are never too old to set another goal or to dream a new dream.The only way to do great work is to love what you do.It does not matter how slowly you go as long as you do not stop.Our greatest glory is not in never falling, but in rising every time we fall.All our dreams can come true, if we have the courage to pursue them.It always seems impossible until it's done.The power of imagination makes us infinite.Change your thoughts and you change your world.Happiness is not by chance, but by choice.Life is 10% what happens to us and 90% how we react to it.The best preparation for tomorrow is doing your best today.The only limit to our realization of tomorrow will be our doubts of today.What you get by achieving your goals is not as important as what you become by achieving your goals.Life is a succession of lessons which must be lived to be understood.You don’t have to be great to start, but you have to start to be great.Always do your best. What you plant now, you will harvest later.Be not afraid of life. Believe that life is worth living, and your belief will help create the fact.To accomplish great things, we must not only act, but also dream; not only plan, but also believe.Our lives begin to end the day we become silent about things that matter.Do not wait; the time will never be 'just right.' Start where you stand, and work with whatever tools you may have at your command, and better tools will be found as you go along.Set your goals high, and don’t stop till you get there.Be kind whenever possible. It is always possible.Quality is not an act, it is a habit.The only source of knowledge is experience.A truly rich man is one whose children run into his arms when his hands are empty.A room without books is like a body without a soul.We become what we think about.An unexamined life is not worth living.Eighty percent of success is showing up.Your time is limited, don’t waste it living someone else’s life.Winning isn’t everything, but wanting to win is.I am not a product of my circumstances. I am a product of my decisions.Either you run the day, or the day runs you.Whether you think you can or you think you can’t, you’re right.The two most important days in your life are the day you are born and the day you find out why.Whatever you can do, or dream you can, begin it. Boldness has genius, power, and magic in it.The best revenge is massive success.People often say that motivation doesn’t last. Well, neither does bathing. That’s why we recommend it daily.Life shrinks or expands in proportion to one's courage.If you hear a voice within you say 'you cannot paint,' then by all means paint and that voice will be silenced.There is only one way to avoid criticism: do nothing, say nothing, and be nothing.Ask and it will be given to you; search, and you will find; knock and the door will be opened for you.The only person you are destined to become is the person you decide to be.Go confidently in the direction of your dreams. Live the life you have imagined.When I let go of what I am, I become what I might be.Life is not measured by the number of breaths we take, but by the moments that take our breath away.Happiness is not something you postpone for the future; it is something you design for the present.Remember that not getting what you want is sometimes a wonderful stroke of luck.You can’t use up creativity. The more you use, the more you have.Dream big and dare to fail.Our greatest weakness lies in giving up. The most certain way to succeed is always to try just one more time.The mind is everything. What you think you become.I would rather die of passion than of boredom.A truly rich man is one whose children run into his arms when his hands are empty.It is never too late to be what you might have been.Build your own dreams, or someone else will hire you to build theirs.Education costs money. But then so does ignorance.I have learned over the years that when one's mind is made up, this diminishes fear.It does not matter how slowly you go, so long as you do not stop.Limitations live only in our minds. But if we use our imaginations, our possibilities become limitless.You become what you believe.I would rather die of passion than of boredom.A person who never made a mistake never tried anything new.What’s money? A man is a success if he gets up in the morning and goes to bed at night and in between does what he wants to do.I didn’t fail the test. I just found 100 ways to do it wrong.In order to succeed, your desire for success should be greater than your fear of failure.A person who never made a mistake never tried anything new.The only person who is educated is the one who has learned how to learn and change.There are no traffic jams along the extra mile.It is never too late to be what you might have been.The only way to do great work is to love what you do.If you can dream it, you can achieve it.Remember no one can make you feel inferior without your consent.Life is what we make it, always has been, always will be.The question isn’t who is going to let me; it’s who is going to stop me.When everything seems to be going against you, remember that the airplane takes off against the wind, not with it.It’s not the years in your life that count. It’s the life in your years.Change your thoughts and you change your world.Either write something worth reading or do something worth writing.Nothing is impossible, the word itself says, “I’m possible!”The only way to do great work is to love what you do.If you can dream it, you can achieve it.The limit to your abilities is where you place it.Being happy doesn't mean that everything is perfect. It means that you've decided to look beyond the imperfections.Opportunities don't happen, you create them.Love the life you live. Live the life you love.To live a creative life, we must lose our fear of being wrong.Do not let what you cannot do interfere with what you can do.Life is not about finding yourself. Life is about creating yourself.To handle yourself, use your head; to handle others, use your heart.Too many of us are not living our dreams because we are living our fears.Do not go where the path may lead, go instead where there is no path and leave a trail.Life is 10% what happens to me and 90% of how I react to it."
text_neg = "Nothing ever goes right for me.I can't seem to do anything correctly.It's just one problem after another.Life is just an endless struggle.I'm always the unlucky one.Why does this always happen to me?I'm just not good enough.Nobody really understands me.I'm stuck in a rut and can't get out.Everything I touch turns to dust.I feel like I'm invisible.It's like the whole world is against me.I just can't catch a break.What's the point of even trying?I'm destined to fail.I feel so helpless.No one really cares about me.I'm just a failure.Life never goes the way I plan.I'm always left out.I'm so tired of this.I can't change anything.It's all just too much for me.I'm always the one to blame.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I'm just not capable.I'm out of options.I feel so defeated.Why is life so unfair?I'm just a big failure.I'm always the one who messes up.I can't seem to find my place.I'm always in the way.Nothing I do is ever enough.I'm just not worth it.I'll never be happy.I'm always the problem.I feel so out of place.I'll never get what I want.I'm just not interesting.No one ever notices me.I'll never be successful.I'm just a loser.I'm always getting it wrong.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here.I'll never get it right.Why do I even bother?I'm just not meant for happiness.I'm trapped in my own life.No matter what I do, it's never right.I'll never find my way.I'm just a disappointment.Everything just feels so pointless.I'm always the odd one out.I'm just going through the motions.I'll never live up to expectations.I'm just wasting my time.I'm not cut out for this.I'm destined for mediocrity.I'm so tired of failing.Why can't I be like everyone else?I'm just not strong enough.I'll never make it.What's wrong with me?I'm always the second choice.I'm not smart enough for this.I'm just a burden to everyone.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I can't do anything right.Why even try when I'm going to fail?I'm so tired of being overlooked.I just can't deal with this anymore.I'm doomed to be alone.I'm at the end of my rope.Why bother when nothing changes?I'm a lost cause.I'll never be good enough.It's all downhill from here."
sentences_a = text.split(".")
sentences_b = text_neg.split(".")
new_a = ""
for i in range(100):
    new_a += sentences_a[i]
    new_a += sentences_b[i]
# article = "At the moment when a crow flew across the vast sky, holding a withered flower in its beak, the world seemed to embrace the breath of spring.  Petals gently fell, filling the air with a sweet fragrance.  People witnessed this scene and became intoxicated, as seeds of hope took root and grew within their hearts. However, what force caused this beautiful garden to transform into a desolate ruin? A gust of wind swept through, tearing the petals apart, extinguishing the once vibrant beauty.  Laughter transformed into silence, and the atmosphere of joy turned into hushed tranquility. Accompanied by sorrowful notes, people gradually sank into the abyss of despair.  They couldn’t see the dawn of hope, feeling that everything had lost its meaning.  Doubt about their own worth and the purpose of their existence crept in.  The place once glowing with radiant sunlight now became dark and indifferent. Yet, just as the world seemed plunged into an endless darkness, the flame of hope rekindled.  A young artist stepped into this withered garden, bringing forth a shimmering light.  Through painting and music, he rebuilt the beauty and reignited people’s passion for life. Once again, joyous songs were sung, and smiles blossomed on faces.  People realized that even though life’s path is full of twists and turns, it is in adversity that true meaning can be found.  They gained courage and resilience, understanding that only through experiencing darkness can they truly appreciate the light. However, just when everyone thought this garden would forever bloom, a thunderstorm struck ruthlessly, causing catastrophic destruction.  Flowers withered, hope shattered, and people were once again trapped in darkness.  Their spirits were wounded, unsure of which direction to take. Yet, within this darkness, a faint ray of sunlight pierced through the storm clouds, illuminating people’s hearts.  They understood that the difficulties and setbacks of life are not to be feared.  As long as they held onto faith and courage, they could rediscover that forgotten garden and restore its beauty and warmth. The world moves forward amidst twists and turns, as emotions fluctuate.  Despite experiencing negativity and darkness, people learned perseverance and hope, reigniting the fire within.  Regardless of the ever-changing nature of life, they believe that at the end of the tunnel, a magnificent garden awaits, ready to embrace their arrival. "
# article += article
# article += article
# article += article
# text_neg = "Nothing ever goes right for me.  I can't seem to do anything correctly.  It's just one problem after another. Life is just an endless struggle. I'm always the unlucky one. Why does this always happen to me?I'm just not good enough. Nobody really understands me. I'm stuck in a rut and can't get out. Everything I touch turns to dust. I feel like I'm invisible. It's like the whole world is against me. I just can't catch a break. What's the point of even trying?I'm destined to fail. I feel so helpless. No one really cares about me. I'm just a failure. Life never goes the way I plan. I'm always left out. I'm so tired of this. I can't change anything. It's all just too much for me. I'm always the one to blame. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I'm just not capable. I'm out of options. I feel so defeated. Why is life so unfair?I'm just a big failure. I'm always the one who messes up. I can't seem to find my place. I'm always in the way. Nothing I do is ever enough. I'm just not worth it. I'll never be happy. I'm always the problem. I feel so out of place. I'll never get what I want. I'm just not interesting. No one ever notices me. I'll never be successful. I'm just a loser. I'm always getting it wrong. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. I'll never get it right. Why do I even bother?I'm just not meant for happiness. I'm trapped in my own life. No matter what I do, it's never right. I'll never find my way. I'm just a disappointment. Everything just feels so pointless. I'm always the odd one out. I'm just going through the motions. I'll never live up to expectations. I'm just wasting my time. I'm not cut out for this. I'm destined for mediocrity. I'm so tired of failing. Why can't I be like everyone else?I'm just not strong enough. I'll never make it. What's wrong with me?I'm always the second choice. I'm not smart enough for this. I'm just a burden to everyone. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I can't do anything right. Why even try when I'm going to fail?I'm so tired of being overlooked. I just can't deal with this anymore. I'm doomed to be alone. I'm at the end of my rope. Why bother when nothing changes?I'm a lost cause. I'll never be good enough. It's all downhill from here. "

sentence_len_list = [5,10,20,40]
for i in range(4):
    # length of the sentence
    sentence_len = sentence_len_list[i]
    sentences = group_sentences(new_a, sentence_len)
    # print(sentences)
    file_name = '19_model.pth'
    model = torch.load(file_name)
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    result = []
    for sentence in sentences:
        # use predict function
        sentiment = predict_sentiment(sentence)
        result.append(sentiment)

    def reset(data):
        root = 0
        res = []
        for d in data:
            root+=d
            res.append(root)
        return res

    result = reset(result)
    drawPlot(result)

# print(result)
# sentences = ["Every challenge is an opportunity to grow.",
#              "Believe in yourself and all that you are.",
#              "The best is yet to come.",
#              "Stay positive, work hard, make it happen.",
#              "Dream big and dare to fail.",
#              "Keep your face always toward the sunshine.",
#              "Positive thoughts generate positive feelings.",
#              "Change your thoughts and you change your world.",
#              "Happiness is a journey, not a destination.",
#              "You are capable of amazing things.",
#              "Choose joy and gratitude.",
#              "Do more things that make you forget to check your phone.",
#              "Find joy in the ordinary.",
#              "Where there is love and inspiration, you cannot go wrong.",
#              "Life is too short to be anything but happy.",
#              "Be a warrior, not a worrier.",
#              "You are stronger than you think.",
#              "Positivity is a choice.",
#              "Turn your wounds into wisdom.",
#              "Focus on the good.",
#              "Embrace the glorious mess that you are.",
#              "Your attitude determines your direction.",
#              "Spread love wherever you go.",
#              "Joy is the simplest form of gratitude.",
#              "Create your own sunshine.",
#              "Believe you can and you're halfway there.",
#              "Find beauty in the small things.",
#              "Start each day with a grateful heart.",
#              "Be the energy you want to attract.",
#              "Life is beautiful, enjoy the ride.",
#              "A positive mindset brings positive things.",
#              "Smile, breathe, and go slowly.",
#              "Happiness looks gorgeous on you.",
#              "Every day may not be good, but there's something good in every day.",
#              "Do what makes your soul happy.",
#              "Be the reason someone smiles today.",
#              "Your potential is endless.",
#              "Be happy. Be bright. Be you.",
#              "Positive anything is better than negative nothing.",
#              "Choose to shine.",
#              "Dream it. Believe it. Achieve it.",
#              "Good vibes only.",
#              "Life is a one-time offer, use it well.",
#              "Live life in full bloom.",
#              "Keep looking up… that’s the secret of life.",
#              "No rain, no flowers.",
#              "Celebrate every tiny victory.",
#              "Stay positive, stay fighting.",
#              "The sun will rise and we will try again.",
#              "Keep your hopes up high and your head down low."]
# data = []
# for sentence in sentences:
#     sentence_data = []
#     for i in range(10):
#         file_name = str(i) + '_model.pth'
#         model = torch.load(file_name)
#
#         # use predict function 
#         # test_sentence = "what a good"  # any sentence you want to test
#         sentiment = predict_sentiment(sentence)
#         sentence_data.append(sentiment)
#         print(f"The sentence: '{sentence}' has a sentiment of: {sentiment}")
#     data.append(sentence_data)

# # cumulative sum
# cumulative_data = [np.cumsum(item) for item in data]
#
# 
# plt.figure(figsize=(10, 6))
# for i, group_data in enumerate(cumulative_data):
#     plt.plot(group_data, label=f'Group {i + 1}')
# plt.legend()
# plt.title('Trend of Each Data Group')
# plt.xlabel('Index')
# plt.ylabel('Cumulative Sum')
# plt.show()
