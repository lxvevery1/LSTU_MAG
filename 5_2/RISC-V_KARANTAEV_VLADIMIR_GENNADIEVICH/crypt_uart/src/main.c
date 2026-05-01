/*
 * AES-256 Key Generation Service (Zephyr RTOS)
 *
 * UART-сервис, принимающий текстовые команды и возвращающий
 * криптографические ключи AES-256 в hex-представлении.
 *
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/device.h>
#include <string.h>
#include <psa/crypto.h>

#define DEV_CONSOLE  DEVICE_DT_GET(DT_CHOSEN(zephyr_console))
#define CMD_BUF_SIZE 64
#define AES256_KEY_BYTES 32  /* 256 бит = 32 байта */

/* ---------- вспомогательные функции UART ---------- */

/* Отправить строку побайтово через UART */
static void uart_send_string(const struct device *dev, const char *str)
{
    while (*str) {
        uart_poll_out(dev, *str++);
    }
}

/* Прочитать строку из UART до символа '\n' или переполнения буфера.
 * Возвращает длину прочитанной строки (без '\n').
 * '\n' заменяется на '\0'. */
static int uart_read_line(const struct device *dev, char *buf, int max_len)
{
    int pos = 0;
    unsigned char c;

    while (pos < max_len - 1) {
        if (uart_poll_in(dev, &c) == 0) {
            if (c == '\n' || c == '\r') {
                break;
            }
            buf[pos++] = (char)c;
        }
    }
    buf[pos] = '\0';
    return pos;
}

/* ---------- криптографические функции ---------- */

/* Инициализация PSA Crypto */
static int crypto_init(void)
{
    psa_status_t status = psa_crypto_init();
    if (status != PSA_SUCCESS) {
        printk("psa_crypto_init() failed: %d\n", status);
        return -1;
    }
    printk("PSA Crypto initialized.\n");
    return 0;
}

/* Генерация случайных 32 байт (AES-256 ключ) через PSA RNG */
static int generate_aes256_key(uint8_t *key_buf)
{
    psa_status_t status = psa_generate_random(key_buf, AES256_KEY_BYTES);
    if (status != PSA_SUCCESS) {
        printk("psa_generate_random() failed: %d\n", status);
        return -1;
    }
    return 0;
}

/* Преобразование массива байт в hex-строку */
static void bytes_to_hex(const uint8_t *src, int src_len,
                         char *dst, int dst_len)
{
    static const char hex_chars[] = "0123456789ABCDEF";
    int i;

    for (i = 0; i < src_len && (i * 2 + 1) < dst_len - 1; i++) {
        dst[i * 2]     = hex_chars[(src[i] >> 4) & 0x0F];
        dst[i * 2 + 1] = hex_chars[src[i] & 0x0F];
    }
    dst[i * 2] = '\0';
}

/* ---------- обработка команд ---------- */

static void handle_command(const struct device *dev, const char *cmd)
{
    if (strcmp(cmd, "GENKEY") == 0) {
        uint8_t key[AES256_KEY_BYTES];
        char hex[AES256_KEY_BYTES * 2 + 1];

        if (generate_aes256_key(key) == 0) {
            bytes_to_hex(key, AES256_KEY_BYTES, hex, sizeof(hex));
            uart_send_string(dev, "KEY:");
            uart_send_string(dev, hex);
            uart_send_string(dev, "\n");
        } else {
            uart_send_string(dev, "ERR:KEYGEN_FAIL\n");
        }
    } else if (strcmp(cmd, "STATUS") == 0) {
        uart_send_string(dev, "OK:READY\n");
    } else if (strcmp(cmd, "EXIT") == 0) {
        uart_send_string(dev, "OK:BYE\n");
    } else {
        uart_send_string(dev, "ERR:UNKNOWN_CMD\n");
    }
}

int main(void)
{
    const struct device *dev = DEV_CONSOLE;

    printk("=== AES-256 Key Generation Service ===\n");

    if (!device_is_ready(dev)) {
        printk("UART device not ready!\n");
        return 1;
    }
    printk("UART ready.\n");

    if (crypto_init() != 0) {
        return 1;
    }

    uart_send_string(dev, "READY\n");

    char cmd_buf[CMD_BUF_SIZE];

    while (1) {
        int len = uart_read_line(dev, cmd_buf, sizeof(cmd_buf));
        if (len > 0) {
            handle_command(dev, cmd_buf);

            if (strcmp(cmd_buf, "EXIT") == 0) {
                break;
            }
        }
    }

    printk("Service stopped.\n");
    return 0;
}
