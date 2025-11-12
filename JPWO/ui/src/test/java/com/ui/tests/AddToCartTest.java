package com.ui.tests;

import com.ui.pages.CartPage;
import com.ui.pages.ContactPage;
import com.ui.pages.LoginPage;
import com.ui.pages.MessagesPage;
import com.ui.pages.ProductPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AddToCartTest {
    private WebDriver driver;
    private ProductPage productPage;
    private CartPage cartPage;
    private LoginPage loginPage;
    private ContactPage contactPage;
    private MessagesPage messagesPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAs("customer3@practicesoftwaretesting.com", "pass123");
        productPage = new ProductPage(driver);
        cartPage = new CartPage(driver);
        contactPage = new ContactPage(driver);
        messagesPage = new MessagesPage(driver);
    }

    @Test
    void shouldAddProductToCart() {
        productPage.open("01K8V39GY1R47X8K89E7DJ0HTX");

        productPage.addToCart();
        assertTrue(productPage.getCartCount() == 1, "Cart should not be empty after adding a product");
        productPage.openCartPage();
        assertTrue(cartPage.containsProduct("Flat-Head Wood Screws"), "Cart does not contain the expected product");
    }

    @Test
    void shouldSendMessage() {
        String uniqueUUID = java.util.UUID.randomUUID().toString();
        String message = uniqueUUID + "text with 50 letters abcdefghijklmnoprstuvwxyzABCD";
        contactPage.open();
        contactPage.selectSubject("Payments");
        contactPage.enterMessage(message);
        contactPage.sendMessage();

        messagesPage.open();
        System.out.println(messagesPage.getMessage(message));
        assertTrue(messagesPage.getMessage(message).contains(uniqueUUID), "The message content does not match");
        assertEquals("NEW", messagesPage.getStatus(message), "The message status is not 'NEW'");
        assertTrue("payments".contains(messagesPage.getSubject(message)), "The message subject is not 'payments'");
    }


    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
