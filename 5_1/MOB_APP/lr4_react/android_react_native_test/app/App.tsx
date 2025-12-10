import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import HomeScreen from './screens/HomeScreen';
import HttpScreen from './screens/HttpScreen';

const Tab = createBottomTabNavigator();

export default function App() {
    return (
        <NavigationContainer>
        <Tab.Navigator>
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="HTTP" component={HttpScreen} />
        </Tab.Navigator>
        </NavigationContainer>
    );
}
