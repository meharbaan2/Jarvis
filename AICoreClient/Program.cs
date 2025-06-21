using Grpc.Net.Client;
using AICoreClient;
using System.Collections.Generic;
using System.Text.Unicode;
using System.Text;

// Punjabi text display
Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

// Step 1: Create a gRPC channel (connect to Python server)
using var channel = GrpcChannel.ForAddress("http://localhost:50051");

// Step 2: Create the gRPC client
var client = new AIService.AIServiceClient(channel);

// Step 3: Continuous interaction loop
while (true)
{
    try
    {
        Console.Write("\nAsk AI (or type 'exit' to quit): ");
        var userInput = Console.ReadLine();

        if (string.IsNullOrWhiteSpace(userInput))
            continue;

        if (userInput.ToLower() == "exit")
            break;

        var request = new QueryRequest { InputText = userInput };
        var reply = client.ProcessQuery(request);

        Console.WriteLine($"\nAI ({reply.AiSource}) says: {reply.ResponseText}\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }
}

Console.WriteLine("Session ended.");