using System;
using System.Windows.Forms;
using Grpc.Net.Client;
using AICoreClient;
using AIAssistantUI;
using Google.Protobuf;
using NAudio.Wave;

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
    private ComboBox languageCombo;
    private Button voiceButton;

    public MainForm()
    {
        this.Text = "J.A.R.V.I.S AI Assistant";
        this.Size = new Size(1000, 700);
        this.BackColor = Color.FromArgb(25, 25, 45);
        this.ForeColor = Color.White;
        this.Font = new Font("Segoe UI", 9);
        this.Padding = new Padding(10);

        InitializeComponents();
        InitializeGrpcClient();
    }

    private void InitializeComponents()
    {
        // Main layout panel
        var mainPanel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
            RowStyles = {
                new RowStyle(SizeType.Percent, 90), // Conversation area
                new RowStyle(SizeType.AutoSize)     // Combined language/input area
            },
            CellBorderStyle = TableLayoutPanelCellBorderStyle.None,
            Padding = new Padding(0)
        };

        // Conversation display
        conversationBox = new RichTextBox
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(15, 15, 30),
            ForeColor = Color.White,
            ReadOnly = true,
            Font = new Font("Segoe UI", 11),
            ScrollBars = RichTextBoxScrollBars.Vertical,
            Margin = new Padding(5),
            BorderStyle = BorderStyle.None
        };

        // Combined panel for language selector and input
        var bottomPanel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(30, 30, 60),
            Padding = new Padding(5)
        };

        // Language selection
        languageCombo = new ComboBox
        {
            Width = 180,
            Height = 28,
            Font = new Font("Segoe UI", 10),
            DropDownStyle = ComboBoxStyle.DropDownList,
            Items = { "English", "Punjabi" },
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(40, 40, 80),
            ForeColor = Color.White,
            Location = new Point(5, 5) // Explicit positioning
        };
        languageCombo.SelectedIndex = 0;
        //languageCombo.Location = new Point(0, 5); // Positioned at top-left of panel
        //languagePanel.Controls.Add(languageCombo);

        // Input panel at bottom
        var inputPanel = new Panel
        {
            Location = new Point(0, 38), // Below language combo
            Height = 50,
            //Size = new Size(this.ClientSize.Width - 10, 50),
            //Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom,
            //BackColor = Color.FromArgb(30, 30, 60)
            Dock = DockStyle.Bottom,
            BackColor = Color.Transparent
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
            Margin = new Padding(0, 5, 140, 0) // Right margin for buttons
        };
        inputBox.KeyDown += InputBox_KeyDown;

        // Voice input button
        voiceButton = new Button
        {
            Dock = DockStyle.Right,
            Text = "Mic",
            Width = 50,
            Font = new Font("Segoe UI", 12),
            BackColor = Color.FromArgb(60, 60, 100),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Margin = new Padding(5, 0, 5, 0)
        };
        voiceButton.FlatAppearance.BorderSize = 0;
        voiceButton.Click += VoiceButton_Click;

        // Send button
        sendButton = new Button
        {
            Dock = DockStyle.Right,
            Text = "Send",
            Width = 80,
            Font = new Font("Segoe UI", 10, FontStyle.Bold),
            BackColor = Color.FromArgb(0, 120, 215),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat
        };
        sendButton.FlatAppearance.BorderSize = 0;
        sendButton.Click += SendButton_Click;

        // Layout
        var inputContainer = new Panel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(0)
        };

        inputContainer.Controls.Add(voiceWave);
        inputContainer.Controls.Add(inputBox);
        inputContainer.Controls.Add(voiceButton);
        inputContainer.Controls.Add(sendButton);

        inputPanel.Controls.Add(inputContainer);

        // Add controls to bottom panel
        bottomPanel.Controls.Add(languageCombo);
        bottomPanel.Controls.Add(inputPanel);

        // Add to main table
        mainPanel.Controls.Add(conversationBox, 0, 0);
        mainPanel.Controls.Add(bottomPanel, 0, 1);

        this.Controls.Add(mainPanel);
    }

    private void InitializeGrpcClient()
    {
        var channel = GrpcChannel.ForAddress("http://localhost:50051");
        client = new AIService.AIServiceClient(channel);
    }

    private async void SendButton_Click(object sender, EventArgs e)
    {
        //if (string.IsNullOrWhiteSpace(inputBox.Text)) return;

        //try
        //{
        //    // Add user message to conversation
        //    AddMessageToConversation("You", inputBox.Text, Color.LightBlue);

        //    // Show voice wave and hide input
        //    inputBox.Visible = false;
        //    voiceWave.Visible = true;
        //    voiceWave.StartSpeaking();

        //    // Send request to server
        //    var request = new QueryRequest { InputText = inputBox.Text };
        //    var reply = await client.ProcessQueryAsync(request);

        //    // Stop voice wave and show input again
        //    voiceWave.StopSpeaking();
        //    voiceWave.Visible = false;
        //    inputBox.Visible = true;

        //    // Add AI response to conversation
        //    AddMessageToConversation("J.A.R.V.I.S", reply.ResponseText, Color.FromArgb(0, 180, 255));

        //    inputBox.Clear();
        //    inputBox.Focus();
        //}
        //catch (Exception ex)
        //{
        //    voiceWave.StopSpeaking();
        //    voiceWave.Visible = false;
        //    inputBox.Visible = true;

        //    AddMessageToConversation("System", $"Error: {ex.Message}", Color.Red);
        //}

        await ProcessInput();

    }

    private async void InputBox_KeyDown(object sender, KeyEventArgs e)
    {
        //if (e.KeyCode == Keys.Enter)
        //{
        //    SendButton_Click(sender, e);
        //    e.SuppressKeyPress = true;
        //}

        if (e.KeyCode == Keys.Enter)
        {
            await ProcessInput();
            e.SuppressKeyPress = true;
        }

    }

    private async void VoiceButton_Click(object sender, EventArgs e)
    {
        try
        {
            inputBox.Visible = false;
            voiceWave.Visible = true;
            voiceWave.StartSpeaking();

            AddMessageToConversation("You", "[Listening...]", Color.LightBlue);

            var audioData = CaptureAudio(5);

            var audioRequest = new AudioRequest
            {
                AudioData = ByteString.CopyFrom(audioData),
                LanguageCode = languageCombo.SelectedIndex == 1 ? "pa-IN" : "en-US",
                SampleRate = 16000
            };

            var textResponse = await client.RecognizeSpeechAsync(audioRequest);
            inputBox.Text = textResponse.ResponseText;

            // Process the recognized text
            await ProcessInput(textResponse.ResponseText);
        }
        catch (Exception ex)
        {
            AddMessageToConversation("System", $"Error: {ex.Message}", Color.Red);
        }
        finally
        {
            voiceWave.StopSpeaking();
            voiceWave.Visible = false;
            inputBox.Visible = true;
        }
    }

    private async Task ProcessInput(string text = null)
    {
        text = text ?? inputBox.Text;
        if (string.IsNullOrWhiteSpace(text)) return;

        try
        {
            AddMessageToConversation("You", text, Color.LightBlue);

            // Show voice wave while processing
            inputBox.Visible = false;
            voiceWave.Visible = true;
            voiceWave.StartSpeaking();

            // Process through AI
            var aiResponse = await client.ProcessQueryAsync(new QueryRequest
            {
                InputText = text
            });

            // Add AI response
            AddMessageToConversation("J.A.R.V.I.S", aiResponse.ResponseText, Color.FromArgb(0, 1, 180, 255));

            inputBox.Clear();
        }
        catch (Exception ex)
        {
            AddMessageToConversation("System", $"Error: {ex.Message}", Color.Red);
        }
        finally
        {
            voiceWave.StopSpeaking();
            voiceWave.Visible = false;
            inputBox.Visible = true;
            inputBox.Focus();
        }
    }

    private byte[] CaptureAudio(int seconds)
    {
        var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(16000, 16, 1)
        };

        var ms = new MemoryStream();
        using (var writer = new WaveFileWriter(ms, waveIn.WaveFormat))
        {
            var stopRecording = new System.Threading.ManualResetEvent(false);
            waveIn.DataAvailable += (s, e) =>
            {
                writer.Write(e.Buffer, 0, e.BytesRecorded);
                if (writer.Length > 16000 * 2 * seconds)
                {
                    stopRecording.Set();
                }
            };

            waveIn.StartRecording();
            stopRecording.WaitOne(TimeSpan.FromSeconds(seconds + 1));
            waveIn.StopRecording();
        }

        return ms.ToArray();
    }

    private void AddMessageToConversation(string sender, string message, Color color)
    {
        conversationBox.SelectionStart = conversationBox.TextLength;
        conversationBox.SelectionColor = color;

        // Format sender name
        conversationBox.SelectionFont = new Font(conversationBox.Font, FontStyle.Bold);
        conversationBox.AppendText($"{sender}: ");

        // Add message
        conversationBox.SelectionFont = conversationBox.Font;
        conversationBox.AppendText(message + "\n\n");

        // Scroll to bottom
        conversationBox.ScrollToCaret();
    }
}