using Grpc.Net.Client;
using AICoreClient;
using System.Collections.Generic;
using System.Text.Unicode;
using System.Text;

// Step 1: Create a gRPC channel (connect to Python server)
using var channel = GrpcChannel.ForAddress("http://localhost:50051");

// Step 2: Create the gRPC client
var client = new AIService.AIServiceClient(channel);

// Step 3: Continuous interaction loop
while (true)
{
    try
    {
        // Set both input and output encodings to UTF - 8
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;

        // Test Punjabi display
        Console.WriteLine("Punjabi Test: ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?");
        Console.Write("\nAsk AI (or type 'exit' to quit): ");
        var userInput = Console.ReadLine();

        if (string.IsNullOrWhiteSpace(userInput))
            continue;

        if (userInput.ToLower() == "exit")
            break;

        var request = new QueryRequest { InputText = userInput };
        var reply = client.ProcessQuery(request);

        Console.OutputEncoding = System.Text.Encoding.UTF8;

        Console.WriteLine($"\nAI ({reply.AiSource}) says: {reply.ResponseText}\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }
}

Console.WriteLine("Session ended.");