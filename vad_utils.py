import os
import webrtcvad
import collections
import contextlib
import sys
import wave
import numpy as np
from utils import wave_function

class VadGenerator():
    
    def __init__(self, clean_speech_ordered_list, frames_per_sec):
        self.clean_speech_ordered_list = clean_speech_ordered_list
        self.frames_per_sec = frames_per_sec
    
    def read_wave(self, path):
        """Reads a .wav file.

        Takes the path, and returns (PCM audio data, sample rate).
        """
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def write_wave(self, path, audio, sample_rate):
        """Writes a .wav file.

        Takes path, PCM audio data, and sample rate.
        """
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        """Filters out non-voiced audio frames.

        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.

        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.

        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.

        Arguments:

        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).

        Returns: A generator that yields PCM audio data.
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False
        voice_seconds = np.zeros((1))
        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            #sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    voice_seconds = np.append(voice_seconds,(ring_buffer[0][0].timestamp))
                    #yield(voice_seconds)
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    voice_seconds = np.append(voice_seconds,(frame.timestamp + frame.duration))
                    triggered = False
                    #yield(voice_seconds)
                    #yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
            voice_seconds = np.append(voice_seconds,(frame.timestamp + frame.duration))
            #yield(voice_seconds)
        #sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        #if voiced_frames:
            #yield b''.join([f.bytes for f in voiced_frames])
            #yield(voice_seconds)

        return(voice_seconds)

    def vad_main(self, sample_number):
        # this function indicates the moments (seconds) when the voice starts and ends
        audio, sample_rate = self.read_wave(self.clean_speech_ordered_list[sample_number])
        # webrtcvad.Vad(int) where int:1,2,3 from least to most agresiveness to select voices frames
        vad = webrtcvad.Vad(2)
        frames = self.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, 30, 300, vad, frames)
        for i, segment in enumerate(segments):
            path = 'chunk-%002d.wav' % (i,)
            #print(' Writing %s' % (path,))
            self.write_wave(path, segment, sample_rate)
        return(segments)

    def vad(self, sample_number):
        # this function converts the output of vad_main function into a (0,1) matrix of the same shape as the stft
        # then it multiplies this matrix by the clean_speech_normalized to obtain the vad_clean_speech_normalized
        voice_seconds = self.vad_main(sample_number)
        voice_frames = voice_seconds * self.frames_per_sec
        stft = wave_function(self.clean_speech_ordered_list[sample_number], 'other')
        zxx_bin = np.zeros((stft.shape[0],stft.shape[1]))
        for j in range(zxx_bin.shape[1]):
            for i in range(1, voice_frames.shape[0]-1, 2):
                if j > voice_frames[i] and j < voice_frames[i+1]:
                    zxx_bin[:,j] = 1             
        return(zxx_bin)
    
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration