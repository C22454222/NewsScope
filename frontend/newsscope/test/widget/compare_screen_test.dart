import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('TabBar renders three tabs', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: DefaultTabController(
          length: 3,
          child: Scaffold(
            appBar: AppBar(
              bottom: const TabBar(
                tabs: [
                  Tab(text: 'Left'),
                  Tab(text: 'Centre'),
                  Tab(text: 'Right'),
                ],
              ),
            ),
            body: const TabBarView(
              children: [
                Center(child: Text('L tab')),
                Center(child: Text('C tab')),
                Center(child: Text('R tab')),
              ],
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Left'), findsOneWidget);
    expect(find.text('Centre'), findsOneWidget);
    expect(find.text('Right'), findsOneWidget);
  });
}
