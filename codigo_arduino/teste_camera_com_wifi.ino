#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ========================================
// DADOS DO WI-FI (JÁ CONFIGURADOS)
// ========================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
// ========================================

WebServer server(80);

// Definição dos pinos para o modelo AI-THINKER ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// --- NOVO: PINO DO FLASH ---
#define FLASH_GPIO_NUM     4  // O LED forte está no GPIO 4
// ---------------------------

// Função que tira a foto e envia para o computador
void handleCapture() {
  Serial.println("Pedido de foto recebido.");

  // --- NOVO: LIGA O FLASH ---
  Serial.println("Ligando Flash...");
  digitalWrite(FLASH_GPIO_NUM, HIGH);
  // Dá um tempo para a câmera se acostumar com a luz forte
  delay(300); 
  // --------------------------

  Serial.println("Capturando...");
  camera_fb_t * fb = NULL;
  
  // Tira a foto
  fb = esp_camera_fb_get();

  // --- NOVO: DESLIGA O FLASH IMEDIATAMENTE ---
  digitalWrite(FLASH_GPIO_NUM, LOW);
  Serial.println("Flash desligado.");
  // ------------------------------------------

  if (!fb) {
    Serial.println("Falha na captura da câmera");
    server.send(500, "text/plain", "Falha na captura");
    return;
  }
  Serial.printf("Foto capturada! Tamanho: %u bytes\n", fb->len);

  // Envia a imagem para o Python
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  WiFiClient client = server.client();
  client.write(fb->buf, fb->len); 

  // LIBERA A MEMÓRIA RAM IMEDIATAMENTE
  esp_camera_fb_return(fb);
  Serial.println("Foto enviada e memoria liberada.");
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // --- NOVO: Configura o pino do flash como saída e garante que começa desligado
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);
  // ----------------------------------------------------------------------------

  // Configuração da Câmera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Resolução: FRAMESIZE_SVGA (800x600) é um bom equilíbrio.
  config.frame_size = FRAMESIZE_SVGA; 
  config.jpeg_quality = 12; // 0-63, menor número = maior qualidade
  config.fb_count = 1; // Usar apenas 1 framebuffer para economizar RAM

  // Inicializa a câmera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro ao iniciar a câmera: 0x%x", err);
    return;
  }

  // Conecta ao Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Conectando ao Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Wi-Fi conectado. IP: ");
  Serial.println(WiFi.localIP());

  // Define a rota que o Python vai chamar
  server.on("/capture", HTTP_GET, handleCapture);
  
  // Inicia o servidor
  server.begin();
  Serial.println("Servidor de câmera pronto!");
  Serial.print("Use este link no Python: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture");
}

void loop() {
  // Mantém o servidor atento a pedidos
  server.handleClient();
}