from typing import Dict, List
import pandas as pd
import numpy as np

class ConfidenceFeatureExtractor:
    """Enhanced version with comprehensive linguistic markers"""
    
    def __init__(self):
        # HEDGING WORDS - 60+ markers
        self.hedging_words = [
            # Basic hedges
            'maybe', 'perhaps', 'probably', 'possibly', 'potentially',
            'might', 'may', 'could', 'would', 'should',
            # Cognitive hedges
            'i think', 'i believe', 'i feel', 'i guess', 'i suppose',
            'i imagine', 'i assume', 'i reckon', 'i suspect',
            # Degree hedges
            'kind of', 'sort of', 'somewhat', 'rather', 'fairly',
            'quite', 'relatively', 'comparatively', 'reasonably',
            'pretty much', 'more or less', 'to some extent',
            # Approximators
            'approximately', 'roughly', 'about', 'around', 'nearly',
            'almost', 'practically', 'virtually', 'essentially',
            # Probability hedges
            'likely', 'unlikely', 'apt to', 'liable to', 'tends to',
            'seems to', 'appears to', 'looks like', 'sounds like',
            # Conditional hedges
            'if possible', 'if available', 'if applicable', 'depending on',
            'subject to', 'provided that', 'assuming that',
            # Minimizers
            'a bit', 'a little', 'slightly', 'marginally',
            'to a degree', 'in a way', 'in a sense',
            # Tentative language
            'tend to', 'seem to', 'appear to', 'suggest that',
            'indicate that', 'imply that',
            # Epistemic hedges
            'as far as i know', 'to my knowledge', 'in my opinion',
            'in my view', 'from my perspective', 'it seems that',
            'it appears that', 'i would say',
            # Additional hedges
            'arguably', 'presumably', 'conceivably', 'theoretically',
            'technically', 'supposedly', 'ostensibly',
            'it could be that', 'it may be that', 'it might be that',
            'one could say', 'it is possible that',
            'more or less', 'give or take',
            'in general', 'in most cases', 'in some cases',
            'at times', 'from time to time',
            'it tends to be', 'it often seems',
        ]
        
        # FILLER WORDS - 50+ markers
        self.filler_words = [
            # Classic fillers
            'um', 'uh', 'er', 'ah', 'eh', 'mm', 'hmm', 'uhh', 'umm',
            # Discourse markers
            'like', 'you know', 'i mean', 'you see',
            'you know what i mean', 'know what i\'m saying',
            # Temporal fillers
            'well', 'so', 'now', 'then', 'anyway', 'anyways',
            'alright', 'okay', 'ok',
            # Intensifiers used as fillers
            'actually', 'basically', 'literally', 'seriously',
            'honestly', 'really', 'totally', 'completely',
            # Repair markers
            'i mean to say', 'what i mean is', 'let me think',
            'how do i put it', 'how should i say',
            # Stalling phrases
            'let me see', 'let me check', 'give me a second',
            'just a moment', 'hold on', 'wait a minute',
            'one second', 'bear with me',
            # Continuation markers
            'and everything', 'and stuff', 'and all that',
            'or something', 'or whatever', 'and so on',
            'et cetera', 'and the like',
            # More fillers
            'right', 'okay then', 'so yeah', 'yeah',
            'well then', 'you know like', 'kind of like',
            'basically speaking', 'to be honest',
            'frankly', 'at the end of the day',
            'by the way', 'on that note',
            'if you will', 'sort of like',
            'just saying', 'that being said',
        ]
        
        # CONFIDENCE WORDS - 70+ markers
        self.confidence_words = [
            # Certainty markers
            'definitely', 'certainly', 'absolutely', 'undoubtedly',
            'without doubt', 'no doubt', 'for sure', 'for certain',
            # Knowledge indicators
            'know', 'knows', 'knew', 'knowing', 'understand',
            'clear', 'clearly', 'obvious', 'obviously', 'evident',
            'evidently', 'apparent', 'apparently',
            # Assurance words
            'sure', 'positive', 'confident', 'certain', 'convinced',
            'assured', 'secure', 'firm',
            # Future commitment
            'will', 'shall', 'going to', 'guarantee', 'guaranteed',
            'promise', 'promised',
            # Strong affirmatives
            'yes', 'correct', 'right', 'exactly', 'precisely',
            'indeed', 'truly', 'surely',
            # Expertise markers
            'expert', 'experienced', 'qualified', 'trained',
            'specialized', 'professional', 'competent',
            # Factual language
            'fact', 'facts', 'proven', 'verified', 'confirmed',
            'established', 'demonstrated', 'shown',
            # Direct statements
            'is', 'are', 'was', 'were', 'been',
            # Solution-oriented
            'solution', 'solve', 'fix', 'resolve', 'handle',
            'address', 'deal with', 'take care of',
            # Immediacy
            'now', 'immediately', 'right away', 'right now',
            'instantly', 'straightaway', 'at once',
            # Strong certainty words
            'unquestionably', 'indisputably', 'irrefutable',
            'categorically', 'decisively', 'conclusively',
            'assuredly', 'unmistakably',
            'validated', 'substantiated',
            'guarantees', 'ensures',
            'commit', 'committed',
            'resolved', 'accomplished',
            'definitive', 'concrete',
            'authoritative', 'credible',
            'dependable', 'reliable',
        ]
        
        # UNCERTAINTY PHRASES - 30+ markers
        self.uncertainty_phrases = [
            # Direct uncertainty
            "i don't know", "i dunno", "i'm not sure", "not sure",
            "i'm uncertain", "i'm unsure", "i have no idea",
            # Confusion
            "i'm confused", "that's confusing", "unclear",
            "i don't understand", "i'm lost",
            # Lack of information
            "i don't have", "i lack", "i'm missing",
            "can't tell", "hard to say", "difficult to say",
            # Inability
            "i can't", "i cannot", "unable to", "not able to",
            "don't know how",
            # Doubt
            "i doubt", "doubtful", "questionable",
            # Need for help
            "need help", "need assistance", "need to check",
            "need to ask", "need to verify",
            # Deferral
            "ask someone else", "check with", "another department",
            # More uncertainty expressions
            "i'm not certain",
            "i'm not entirely sure",
            "i can't confirm",
            "i can't verify",
            "not entirely clear",
            "it depends",
            "it varies",
            "uncertain at this point",
            "still checking",
            "pending confirmation",
            "awaiting information",
            "i'm guessing",
            "i'm speculating",
        ]
        
        # ASSERTIVE WORDS - 25+ markers
        self.assertive_words = [
            'must', 'have to', 'need to', 'got to', 'gotta',
            'ought to', 'should', 'shall', 'will',
            'require', 'required', 'requires', 'necessary',
            'essential', 'crucial', 'critical', 'vital',
            'important', 'imperative', 'mandatory',
            'ensure', 'make sure', 'guarantee',
            # Strong directives
            'cannot', 'must not',
            'obligated', 'compulsory',
            'non-negotiable', 'enforced',
            'strictly', 'demand',
            'insist', 'direct',
            'authorize', 'approved',
            'enforce', 'implement',
        ]
        
        # POLITENESS MARKERS - 20+ markers
        self.politeness_markers = [
            'sorry', 'apologize', 'apologies', 'excuse me',
            'pardon', 'please', 'kindly', 'thank you',
            'thanks', 'appreciate', 'grateful',
            'would you mind', 'could you please', 'if possible',
            'with respect', 'respectfully', 'if i may',
            # Additional politeness phrases
            'much appreciated',
            'thank you very much',
            'i appreciate your patience',
            'i appreciate your understanding',
            'my pleasure',
            'happy to help',
            'feel free',
            'at your convenience',
            'whenever you’re ready',
        ]
        
        # ABSOLUTE WORDS - 30+ markers
        self.absolute_words = [
            'all', 'every', 'everyone', 'everybody', 'everything',
            'always', 'forever', 'constantly', 'continually',
            'none', 'nothing', 'nobody', 'no one', 'nowhere',
            'never', 'complete', 'completely', 'total', 'totally',
            'entire', 'entirely', 'whole', 'wholly',
            'full', 'fully', 'absolute', 'absolutely',
            # Strong absolutes
            'definitively',
            'comprehensively',
            'universally',
            'unconditionally',
            'irrevocably',
            'permanently',
            'entirety',
            'in every case',
            'without exception',
        ]
        
        # QUESTION WORDS - 15+ markers
        self.question_words = [
            'what', 'when', 'where', 'why', 'who', 'whom',
            'which', 'whose', 'how',
            'is it', 'are there', 'do you', 'can you',
            'could you', 'would you',
            # More interrogatives
            'does it',
            'did you',
            'will you',
            'shall we',
            'why is',
            'how come',
            'what if',
            'who else',
        ]
        
        # POSITIVE EMOTIONS - 13 markers
        self.positive_emotion_words = [
            'happy', 'glad', 'pleased', 'delighted', 'excited',
            'confident', 'proud', 'satisfied', 'content',
            'great', 'excellent', 'perfect', 'wonderful',
            # More positive emotion terms
            'thrilled', 'joyful', 'optimistic',
            'relieved', 'grateful', 'encouraged',
            'motivated', 'inspired',
            'hopeful', 'enthusiastic',
            'fantastic', 'amazing',
        ]
        
        # NEGATIVE EMOTIONS - 12 markers
        self.negative_emotion_words = [
            'worried', 'concerned', 'anxious', 'nervous',
            'hesitant', 'reluctant', 'uncomfortable', 'uneasy',
            'stressed', 'frustrated', 'confused', 'unsure',
            # More negative emotion terms
            'upset', 'disappointed',
            'overwhelmed', 'irritated',
            'annoyed', 'discouraged',
            'hopeless', 'dissatisfied',
            'regretful', 'angry',
        ]
        
        # ESCALATION PHRASES - 9 markers
        self.escalation_phrases = [
            'transfer you', 'transfer to', 'supervisor',
            'manager', 'specialist', 'someone who can',
            'better suited', 'another department', 'escalate',
            # More escalation markers
            'let me transfer',
            'connect you with',
            'pass this to',
            'refer you to',
            'forward this to',
            'escalate this matter',
            'bring this to the attention of',
        ]
        
        # OWNERSHIP PHRASES - 10 markers
        self.ownership_phrases = [
            'i will', 'i can', "i'll handle", "i'll take care",
            'let me help', 'i can help', "i'll assist",
            "i'll resolve", "i'll fix", "i'll solve",
            # Strong ownership language
            'i’ll personally',
            'i’ll make sure',
            'i’ll ensure',
            'i’ll follow through',
            'leave it with me',
            'i’ve got this',
            'i’ll handle this for you',
        ]
        
        # DELAY PHRASES - 9 markers
        self.delay_phrases = [
            'call back', 'get back to you', 'follow up',
            'check on that', 'look into', 'investigate',
            'find out', 'research', 'verify',
            # More delay markers
            'get back shortly',
            'revert back',
            'circle back',
            'update you',
            'keep you posted',
            'await response',
            'pending review',
            'in progress',
        ]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive confidence features"""
        text_lower = str(text).lower()
        words = text_lower.split()
        word_count = max(len(words), 1)
        
        features = {}
        
        # Basic ratios
        features['hedging_ratio'] = self._count_matches(text_lower, self.hedging_words) / word_count
        features['filler_ratio'] = self._count_matches(text_lower, self.filler_words) / word_count
        features['confidence_ratio'] = self._count_matches(text_lower, self.confidence_words) / word_count
        features['assertive_ratio'] = self._count_matches(text_lower, self.assertive_words) / word_count
        features['politeness_ratio'] = self._count_matches(text_lower, self.politeness_markers) / word_count
        features['absolute_ratio'] = self._count_matches(text_lower, self.absolute_words) / word_count
        
        # Phrase counts
        features['uncertainty_count'] = self._count_matches(text_lower, self.uncertainty_phrases)
        features['question_word_count'] = self._count_matches(text_lower, self.question_words)
        
        # Emotional markers
        features['positive_emotion_ratio'] = self._count_matches(text_lower, self.positive_emotion_words) / word_count
        features['negative_emotion_ratio'] = self._count_matches(text_lower, self.negative_emotion_words) / word_count
        
        # Call center specific
        features['escalation_count'] = self._count_matches(text_lower, self.escalation_phrases)
        features['ownership_count'] = self._count_matches(text_lower, self.ownership_phrases)
        features['delay_count'] = self._count_matches(text_lower, self.delay_phrases)
        
        # Punctuation
        features['question_ratio'] = text.count('?') / word_count
        features['exclamation_ratio'] = text.count('!') / word_count
        
        # Sentence structure
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_length'] = word_count
        
        # First person
        first_person = ['i ', 'me ', 'my ', 'mine ', 'myself ']
        features['first_person_ratio'] = sum(1 for fp in first_person if fp in text_lower + ' ') / word_count
        
        # Statement type
        features['is_statement'] = 1 if not text.strip().endswith('?') else 0
        
        # Composite score
        features['confidence_score'] = (
            features['confidence_ratio'] + 
            features['assertive_ratio'] +
            features['ownership_count'] / word_count -
            features['hedging_ratio'] -
            features['filler_ratio'] -
            features['uncertainty_count'] / word_count
        )
        
        return features
    
    def _count_matches(self, text: str, word_list: List[str]) -> int:
        """Count matches in text"""
        return sum(1 for item in word_list if item in text)
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extract features for multiple texts"""
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all markers"""
        summary_data = [
            {'Category': 'Hedging Words', 'Count': len(self.hedging_words)},
            {'Category': 'Filler Words', 'Count': len(self.filler_words)},
            {'Category': 'Confidence Words', 'Count': len(self.confidence_words)},
            {'Category': 'Uncertainty Phrases', 'Count': len(self.uncertainty_phrases)},
            {'Category': 'Assertive Words', 'Count': len(self.assertive_words)},
            {'Category': 'Politeness Markers', 'Count': len(self.politeness_markers)},
            {'Category': 'Absolute Words', 'Count': len(self.absolute_words)},
            {'Category': 'Question Words', 'Count': len(self.question_words)},
            {'Category': 'Positive Emotions', 'Count': len(self.positive_emotion_words)},
            {'Category': 'Negative Emotions', 'Count': len(self.negative_emotion_words)},
            {'Category': 'Escalation Phrases', 'Count': len(self.escalation_phrases)},
            {'Category': 'Ownership Phrases', 'Count': len(self.ownership_phrases)},
            {'Category': 'Delay Phrases', 'Count': len(self.delay_phrases)},
        ]
        return pd.DataFrame(summary_data)