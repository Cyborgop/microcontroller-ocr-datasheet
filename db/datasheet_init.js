db = db.getSiblingDB('microcontrollers');

db.datasheets.insertMany([
  {
    part_number: "LPC1768",
    manufacturer: "NXP Semiconductors",
    core: "ARM Cortex-M3",
    flash: "512KB",
    ram: "64KB",
    max_clock: "100MHz",
    datasheet_url: "https://www.nxp.com/docs/en/data-sheet/LPC1768_66.pdf",
    features: [
      "32-bit MCU",
      "Ethernet, USB, CAN",
      "Multiple UARTs, ADC, DAC",
      "Timers, PWM, I2C, SPI"
    ]
  },
  {
    part_number: "8051",
    manufacturer: "Intel (original), many vendors",
    core: "8051 (MCS-51)",
    flash: "Varies (typically 4KB–64KB)",
    ram: "128B–4KB",
    max_clock: "12–40MHz",
    datasheet_url: "https://www.ti.com/lit/ds/symlink/mcs51.pdf",
    features: [
      "8-bit MCU",
      "Harvard architecture",
      "UART, Timers, Interrupts",
      "Widely used in education and industry"
    ]
  },
  {
    part_number: "ATmega328P",
    manufacturer: "Microchip (Atmel)",
    core: "AVR 8-bit",
    flash: "32KB",
    ram: "2KB",
    max_clock: "20MHz",
    datasheet_url: "https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf",
    features: [
      "8-bit AVR RISC architecture",
      "23 I/O lines",
      "10-bit ADC",
      "6 PWM channels",
      "Used in Arduino Uno"
    ]
  },
  {
    part_number: "ARM Cortex-M3",
    manufacturer: "ARM",
    core: "ARM Cortex-M3",
    flash: "Varies",
    ram: "Varies",
    max_clock: "Varies",
    datasheet_url: "https://developer.arm.com/documentation/dui0553/latest/",
    features: [
      "32-bit MCU architecture",
      "Widely used in embedded systems",
      "Low power consumption",
      "High performance"
    ]
  },
  {
    part_number: "ARM Cortex-M7",
    manufacturer: "ARM",
    core: "ARM Cortex-M7",
    flash: "Varies",
    ram: "Varies",
    max_clock: "Varies",
    datasheet_url: "https://developer.arm.com/documentation/dui0553/latest/",
    features: [
      "High performance 32-bit MCU",
      "DSP and FPU support",
      "Used in advanced embedded applications"
    ]
  },
  {
    part_number: "ESP32 DevKit",
    manufacturer: "Espressif",
    core: "Xtensa dual-core 32-bit",
    flash: "4MB",
    ram: "520KB SRAM",
    max_clock: "240MHz",
    datasheet_url: "https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf",
    features: [
      "Wi-Fi and Bluetooth",
      "Low power consumption",
      "Multiple GPIOs",
      "ADC, DAC, SPI, I2C"
    ]
  },
  {
    part_number: "NodeMCU ESP8266",
    manufacturer: "Espressif",
    core: "Xtensa 32-bit",
    flash: "4MB",
    ram: "160KB SRAM",
    max_clock: "80MHz",
    datasheet_url: "https://www.espressif.com/sites/default/files/documentation/0a-esp8266ex_datasheet_en.pdf",
    features: [
      "Wi-Fi",
      "Low power consumption",
      "GPIOs, ADC, SPI, I2C",
      "Used in IoT applications"
    ]
  },
  {
    part_number: "Raspberry Pi 3B Plus",
    manufacturer: "Raspberry Pi Foundation",
    core: "ARM Cortex-A53",
    flash: "MicroSD",
    ram: "1GB LPDDR2",
    max_clock: "1.4GHz",
    datasheet_url: "https://www.raspberrypi.com/documentation/computers/raspberry-pi.html",
    features: [
      "Broadcom BCM2837B0",
      "Wi-Fi, Bluetooth",
      "Multiple USB ports",
      "HDMI output"
    ]
  }
]);
