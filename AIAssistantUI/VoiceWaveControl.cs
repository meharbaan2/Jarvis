using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace AIAssistantUI
{
    public class VoiceWaveControl : Control  // Changed from UserControl to Control
    {
        private System.Windows.Forms.Timer animationTimer;
        private readonly Random random = new Random();
        private float[] waveData = new float[100];
        private bool isSpeaking = false;
        private Color waveColor = Color.FromArgb(0, 120, 215);

        public VoiceWaveControl()
        {
            // Basic control setup
            SetStyle(ControlStyles.OptimizedDoubleBuffer |
                   ControlStyles.ResizeRedraw |
                   ControlStyles.AllPaintingInWmPaint |
                   ControlStyles.UserPaint |
                   ControlStyles.SupportsTransparentBackColor, true);

            this.DoubleBuffered = true;
            this.BackColor = Color.Transparent;

            // Timer setup
            animationTimer = new System.Windows.Forms.Timer();
            animationTimer.Interval = 50;
            animationTimer.Tick += AnimationTimer_Tick;

            // Initialize wave data
            for (int i = 0; i < waveData.Length; i++)
            {
                waveData[i] = 0;
            }
        }

        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= 0x20; // WS_EX_TRANSPARENT
                return cp;
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                animationTimer?.Stop();
                animationTimer?.Dispose();
            }
            base.Dispose(disposing);
        }

        public void StartSpeaking()
        {
            isSpeaking = true;
            animationTimer.Start();
        }

        public void StopSpeaking()
        {
            isSpeaking = false;
            animationTimer.Stop();
            this.Invalidate();
        }

        private void AnimationTimer_Tick(object sender, EventArgs e)
        {
            if (!isSpeaking) return;

            for (int i = 0; i < waveData.Length; i++)
            {
                waveData[i] = waveData[i] * 0.8f + (random.NextFloat(-1, 1) * 0.2f);
            }

            this.Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            if (!isSpeaking) return;

            Graphics g = e.Graphics;
            int width = this.Width;
            int height = this.Height;
            int centerY = height / 2;

            using (var bgBrush = new LinearGradientBrush(
                this.ClientRectangle,
                Color.FromArgb(20, 0, 40, 80),
                Color.FromArgb(80, 0, 20, 40),
                90f))
            {
                g.FillRectangle(bgBrush, this.ClientRectangle);
            }

            PointF[] points = new PointF[waveData.Length];
            float xStep = width / (float)(waveData.Length - 1);

            for (int i = 0; i < waveData.Length; i++)
            {
                float x = i * xStep;
                float y = centerY + (waveData[i] * centerY * 0.8f);
                points[i] = new PointF(x, y);
            }

            using (var wavePen = new Pen(waveColor, 2))
            using (var fillBrush = new SolidBrush(Color.FromArgb(60, waveColor)))
            using (var path = new GraphicsPath())
            {
                path.AddCurve(points);
                path.AddLine(points[points.Length - 1], new PointF(width, height));
                path.AddLine(new PointF(width, height), new PointF(0, height));
                path.AddLine(new PointF(0, height), points[0]);

                g.FillPath(fillBrush, path);
                g.DrawCurve(wavePen, points);
            }

            for (int i = 0; i < 10; i++)
            {
                int idx = random.Next(waveData.Length);
                float x = idx * xStep;
                float y = centerY + (waveData[idx] * centerY * 0.8f);

                using (var particleBrush = new SolidBrush(Color.FromArgb(random.Next(100, 200), waveColor)))
                {
                    float size = random.Next(2, 6);
                    g.FillEllipse(particleBrush, x - size / 2, y - size / 2, size, size);
                }
            }
        }
    }

    public static class RandomExtensions
    {
        public static float NextFloat(this Random random, float min, float max)
        {
            return (float)(random.NextDouble() * (max - min) + min);
        }
    }
}