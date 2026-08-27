import 'package:flutter/material.dart';

import '../state/app_state.dart';
import 'coach_screen.dart';
import 'exercise_screen.dart';
import 'history_screen.dart';
import 'profile_screen.dart';


class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  final AppState _state = AppState();
  int _tab = 0;

  @override
  void initState() {
    super.initState();
    _state.load();
  }

  @override
  void dispose() {
    _state.dispose();
    super.dispose();
  }

  static const _titles = ['Вправи', 'Історія', 'Тренер', 'Профіль'];

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _state,
      builder: (context, _) {
        final pages = [
          ExerciseScreen(state: _state),
          HistoryScreen(state: _state),
          CoachScreen(state: _state),
          ProfileScreen(
            key: ValueKey(_state.profile.serviceUrl + _state.profile.side),
            state: _state,
          ),
        ];

        return Scaffold(
          appBar: AppBar(
            title: Text(_titles[_tab]),
            actions: [
              if (_tab == 0 || _tab == 2)
                IconButton(
                  onPressed: _state.connect,
                  icon: const Icon(Icons.refresh),
                  tooltip: 'Оновити зв\'язок',
                ),
            ],
          ),
          body: pages[_tab],
          bottomNavigationBar: NavigationBar(
            selectedIndex: _tab,
            onDestinationSelected: (index) => setState(() => _tab = index),
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.fitness_center_outlined),
                selectedIcon: Icon(Icons.fitness_center),
                label: 'Вправи',
              ),
              NavigationDestination(
                icon: Icon(Icons.history_outlined),
                selectedIcon: Icon(Icons.history),
                label: 'Історія',
              ),
              NavigationDestination(
                icon: Icon(Icons.auto_awesome_outlined),
                selectedIcon: Icon(Icons.auto_awesome),
                label: 'Тренер',
              ),
              NavigationDestination(
                icon: Icon(Icons.person_outline),
                selectedIcon: Icon(Icons.person),
                label: 'Профіль',
              ),
            ],
          ),
        );
      },
    );
  }
}
