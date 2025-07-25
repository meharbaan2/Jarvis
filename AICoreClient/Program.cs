using Grpc.Net.Client;
using AICoreClient;
using System;
using System.IO;
using System.Speech.Recognition;
using Google.Protobuf;

// Punjabi text support
Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.InputEncoding = System.Text.Encoding.UTF8;

// gRPC setup
using var channel = GrpcChannel.ForAddress("http://localhost:50051");
var client = new AIService.AIServiceClient(channel);

// Audio capture function using NAudio (better than System.Speech for raw capture)
static byte[] CaptureAudio(int seconds = 5)
{
    // Install-Package NAudio
    using var waveIn = new NAudio.Wave.WaveInEvent
    {
        WaveFormat = new NAudio.Wave.WaveFormat(16000, 16, 1) // 16kHz, 16-bit, mono
    };

    var ms = new MemoryStream();
    using (var writer = new NAudio.Wave.WaveFileWriter(ms, waveIn.WaveFormat))
    {
        var stopRecording = new System.Threading.ManualResetEvent(false);
        waveIn.DataAvailable += (s, e) =>
        {
            writer.Write(e.Buffer, 0, e.BytesRecorded);
            if (writer.Length > 16000 * 2 * seconds) // 16-bit * seconds
            {
                stopRecording.Set();
            }
        };

        waveIn.StartRecording();
        stopRecording.WaitOne(TimeSpan.FromSeconds(seconds + 1)); // Max timeout
        waveIn.StopRecording();
    }

    return ms.ToArray();
}

// Main loop
while (true)
{
    try
    {
        Console.WriteLine("\nChoose input method:");
        Console.WriteLine("1. Type text");
        Console.WriteLine("2. Speak (5 second max)");
        Console.WriteLine("3. Exit");
        Console.Write("> ");

        var choice = Console.ReadLine();
        string textInput = "";

        if (choice == "2") // Voice input
        {
            Console.WriteLine("\nSpeak now... (5 second maximum)");
            var audioData = CaptureAudio();

            // Send audio to server
            var audioRequest = new AudioRequest
            {
                AudioData = ByteString.CopyFrom(audioData),
                LanguageCode = "en-US", // or "pa-IN"
                SampleRate = 16000
            };

            var textResponse = client.RecognizeSpeech(audioRequest);
            textInput = textResponse.ResponseText;

            Console.WriteLine($"\nYou said: {textInput}");
        }
        else if (choice == "1") // Text input
        {
            Console.Write("\nEnter your message: ");
            textInput = Console.ReadLine() ?? "";
        }
        else if (choice == "3")
        {
            break;
        }

        if (string.IsNullOrWhiteSpace(textInput)) continue;

        // Process through AI
        var aiResponse = client.ProcessQuery(new QueryRequest
        {
            InputText = textInput
        });

        Console.WriteLine($"\nAI ({aiResponse.AiSource}) says: {aiResponse.ResponseText}\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }
}