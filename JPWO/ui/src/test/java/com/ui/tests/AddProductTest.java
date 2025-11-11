package com.ui.tests;

import com.ui.pages.AddBrandPage;
import com.ui.pages.AddCategoryPage;
import com.ui.pages.AddProductPage;
import com.ui.pages.BrandAdminPage;
import com.ui.pages.CategoryAdminPage;
import com.ui.pages.LoginPage;
import com.ui.pages.ProductAdminPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddProductTest {

    private WebDriver driver;
    private LoginPage loginPage;
    private AddProductPage addProductPage;
    private ProductAdminPage productAdminPage;
    private AddCategoryPage addCategoryPage;
    private CategoryAdminPage categoryAdminPage;
    private AddBrandPage addBrandPage;
    private BrandAdminPage brandAdminPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAsAdmin("admin@practicesoftwaretesting.com", "welcome01");
        addProductPage = new AddProductPage(driver);
        productAdminPage = new ProductAdminPage(driver);
        addCategoryPage = new AddCategoryPage(driver);
        categoryAdminPage = new CategoryAdminPage(driver);
        addBrandPage = new AddBrandPage(driver);
        brandAdminPage = new BrandAdminPage(driver);
    }

    @Test
    void shouldAddNewProduct() throws InterruptedException {
        String uniqueProductName = "New Product " + System.currentTimeMillis();
        addProductPage.open();
        addProductPage.fillForm(uniqueProductName, "New product's description", 100, 29.99,
                "ForgeFlex Tools", "Hand Tools", "Hammer");
        productAdminPage.open();
        productAdminPage.search(uniqueProductName);
        String addedProductName = productAdminPage.getProductName(uniqueProductName);
        String addedProductPrice = productAdminPage.getProductPrice(uniqueProductName);

        assertEquals(uniqueProductName, addedProductName);
        assertEquals("$29.99", addedProductPrice);
    }

    @Test
    void shouldAddNewCategory() {
        String uniqueCategoryName = "New Category " + System.currentTimeMillis();
        String uniqueSlug = "new-category-" + System.currentTimeMillis();
        addCategoryPage.open();
        addCategoryPage.fillForm(uniqueCategoryName, uniqueSlug);
        categoryAdminPage.open();
        categoryAdminPage.search(uniqueCategoryName);
        String addedCategoryName = categoryAdminPage.getCategoryName(uniqueCategoryName);
        String addedCategorySlug = categoryAdminPage.getCategorySlug(uniqueCategoryName);
        assertEquals(uniqueCategoryName, addedCategoryName);
        assertEquals(uniqueSlug, addedCategorySlug);
    }

    @Test
    void shouldAddNewBrand() {
        String uniqueBrandName = "New Brand " + System.currentTimeMillis();
        String uniqueSlug = "new-brand-" + System.currentTimeMillis();
        addBrandPage.open();
        addBrandPage.fillForm(uniqueBrandName, uniqueSlug);
        brandAdminPage.open();
        brandAdminPage.search(uniqueBrandName);
        String addedBrandName = brandAdminPage.getBrandName(uniqueBrandName);
        String addedBrandSlug = brandAdminPage.getBrandSlug(uniqueBrandName);
        assertEquals(uniqueBrandName, addedBrandName);
        assertEquals(uniqueSlug, addedBrandSlug);
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
