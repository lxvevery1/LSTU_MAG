/**
 * Metro configuration for React Native
 * https://github.com/facebook/metro
 *
 * @format
 */

const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

config.transformer.unstable_allowRequireContext = true;

module.exports = config;
