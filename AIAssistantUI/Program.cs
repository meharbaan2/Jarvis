using System;
using System.Windows.Forms;
using Grpc.Net.Client;
using AICoreClient;
using AIAssistantUI;

static class Program
{
    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new MainForm());
    }
}

public class MainForm : Form
{
    private AIService.AIServiceClient client;
    private TextBox inputBox;
    private Button sendButton;
    private VoiceWaveControl voiceWave;
    private RichTextBox conversationBox;

    public MainForm()
    {
        this.Text = "J.A.R.V.I.S AI Assistant";
        this.Size = new Size(800, 600);
        this.BackColor = Color.FromArgb(20, 20, 40);
        this.ForeColor = Color.White;

        InitializeComponents();
        InitializeGrpcClient();
    }

    private void InitializeComponents()
    {
        // Conversation display
        conversationBox = new RichTextBox
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(10, 10, 20),
            ForeColor = Color.White,
            ReadOnly = true,
            Font = new Font("Segoe UI", 11),
            Margin = new Padding(10)
        };

        // Input panel at bottom
        var inputPanel = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 80,
            BackColor = Color.FromArgb(30, 30, 60)
        };

        // Voice wave visualization
        voiceWave = new VoiceWaveControl
        {
            Dock = DockStyle.Fill,
            Visible = false
        };

        // Input box
        inputBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Font = new Font("Segoe UI", 12),
            BackColor = Color.FromArgb(40, 40, 80),
            ForeColor = Color.White,
            Margin = new Padding(10, 10, 5, 10)
        };
        inputBox.KeyDown += InputBox_KeyDown;

        // Send button
        sendButton = new Button
        {
            Dock = DockStyle.Right,
            Text = "Send",
            Width = 80,
            Font = new Font("Segoe UI", 10, FontStyle.Bold),
            BackColor = Color.FromArgb(0, 120, 215),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Margin = new Padding(5, 10, 10, 10)
        };
        sendButton.FlatAppearance.BorderSize = 0;
        sendButton.Click += SendButton_Click;

        // Layout
        inputPanel.Controls.Add(voiceWave);
        inputPanel.Controls.Add(inputBox);
        inputPanel.Controls.Add(sendButton);

        this.Controls.Add(conversationBox);
        this.Controls.Add(inputPanel);
    }

    private void InitializeGrpcClient()
    {
        var channel = GrpcChannel.ForAddress("http://localhost:50051");
        client = new AIService.AIServiceClient(channel);
    }

    private async void SendButton_Click(object sender, EventArgs e)
    {
        if (string.IsNullOrWhiteSpace(inputBox.Text)) return;

        try
        {
            // Add user message to conversation
            AddMessageToConversation("You", inputBox.Text, Color.LightBlue);

            // Show voice wave and hide input
            inputBox.Visible = false;
            voiceWave.Visible = true;
            voiceWave.StartSpeaking();

            // Send request to server
            var request = new QueryRequest { InputText = inputBox.Text };
            var reply = await client.ProcessQueryAsync(request);

            // Stop voice wave and show input again
            voiceWave.StopSpeaking();
            voiceWave.Visible = false;
            inputBox.Visible = true;

            // Add AI response to conversation
            AddMessageToConversation("J.A.R.V.I.S", reply.ResponseText, Color.FromArgb(0, 180, 255));

            inputBox.Clear();
            inputBox.Focus();
        }
        catch (Exception ex)
        {
            voiceWave.StopSpeaking();
            voiceWave.Visible = false;
            inputBox.Visible = true;

            AddMessageToConversation("System", $"Error: {ex.Message}", Color.Red);
        }
    }

    private void InputBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Enter)
        {
            SendButton_Click(sender, e);
            e.SuppressKeyPress = true;
        }
    }

    private void AddMessageToConversation(string sender, string message, Color color)
    {
        conversationBox.SelectionStart = conversationBox.TextLength;
        conversationBox.SelectionColor = color;

        // Format sender name
        conversationBox.AppendText($"{sender}: ");
        conversationBox.SelectionFont = new Font(conversationBox.Font, FontStyle.Bold);

        // Add message
        conversationBox.AppendText(message + "\n\n");
        conversationBox.SelectionFont = conversationBox.Font;

        // Scroll to bottom
        conversationBox.ScrollToCaret();
    }
}